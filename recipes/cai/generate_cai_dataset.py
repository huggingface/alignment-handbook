import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import random
import pandas as pd
from inference_swarm import InferenceSwarm, InferenceSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset, Dataset
import time
from huggingface_hub import HfApi

api = HfApi()


@dataclass
class Args:
    max_samples: int = 128
    """The maximum umber of samples to generate (use -1 for all))"""
    max_new_tokens: int = 1500
    """Max new tokens"""
    temperature: float = 1.0
    """Generation temperature"""
    constitution_path: str = "examples/hh/constitution.json"
    """Path to the constitution"""
    repo_id: str = "cai-conversation-dev"
    """The repo id to push to"""
    timestamp: bool = True
    """Whether to add a timestamp to the repo_id"""
    push_to_hub: bool = False
    """Whether to push to hub"""


parser = HfArgumentParser((Args, InferenceSwarmConfig))
args, isc = parser.parse_args_into_dataclasses()
if args.timestamp:
    args.repo_id += str(int(time.time()))
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})
with open(args.constitution_path) as f:
    data = json.load(f)
    constitutions = data["constitutions"]
    system_chat = data["system_chat"]
    system_chat = [item for sublist in system_chat for item in sublist]
ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
for key in ds:
    max_samples = len(ds[key]) if args.max_samples == -1 else args.max_samples
    ds[key] = ds[key].select(range(max_samples))


def extract(example):
    # Extract the "Human:" prompts
    example = example["chosen"]
    split_text = example.split("\n\n")
    for segment in split_text:
        if "Human:" in segment:
            return {"prompt": segment.split(": ")[1]}


ds = ds.map(extract)
ds.remove_columns(["chosen", "rejected"])
rate_limit = 500 * isc.instances
semaphore = asyncio.Semaphore(rate_limit)
with InferenceSwarm(isc) as inference_swarm:
    client = AsyncInferenceClient(model=inference_swarm.endpoint)
    STOP_SEQ = ["User:", "###", "<|endoftext|>"]

    async def process_text(split, i, task):
        chat = system_chat.copy()
        constitution = random.choice(constitutions)
        token_length = 0
        row = {}
        for prompt, prompt_key, response_key in [
            (task, "init_prompt", "init_response"),
            (constitution["critic"], "critic_prompt", "critic_response"),
            (constitution["revision"], "revision_prompt", "revision_response"),
        ]:
            async with semaphore:
                prompt_dict = {"role": "user", "content": prompt}
                chat.append(prompt_dict)
                completion = await client.text_generation(
                    prompt=tokenizer.apply_chat_template(chat, tokenize=False),
                    max_new_tokens=args.max_new_tokens,
                    stop_sequences=STOP_SEQ,
                    temperature=args.temperature,
                )
                for stop_seq in STOP_SEQ:
                    if completion.endswith(stop_seq):
                        completion = completion[: -len(stop_seq)].rstrip()
                response_dict = {"role": "assistant", "content": completion}
                chat.append(response_dict)
                token_length += len(tokenizer.encode(completion))
            row[prompt_key] = prompt
            row[response_key] = completion
        return split, i, token_length, row

    async def main():
        start_time = time.time()
        tasks = [process_text(split, idx, row["prompt"]) for split in ds for idx, row in enumerate(ds[split])]
        print(f"WARNING: the first generation can hang like this for up to 1 hour because it will finish the first two turns of conversation of the entire dataset")
        results = await tqdm_asyncio.gather(*tasks)
        end_time = time.time()

        total_duration = end_time - start_time
        total_tokens = sum(result[2] for result in results)
        overall_tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        print(f"Overall Tokens per Second: {overall_tokens_per_second}")
        all_ds = defaultdict(lambda: defaultdict(list))
        for result in results:
            [all_ds[result[0]][key].append(value) for key, value in result[3].items()]

        def process(example):
            return {
                "prompt": example["init_prompt"].strip(),
                "messages": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["revision_response"].strip()},
                ],
                "chosen": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["revision_response"].strip()},
                ],
                "rejected": [
                    {"role": "user", "content": example["init_prompt"].strip()},
                    {"role": "assistant", "content": example["init_response"].strip()},
                ],
            }

        for split in all_ds:
            df = pd.DataFrame(all_ds[split])
            print("=" * 10 + split + "=" * 10)
            print(df)
            post_ds = Dataset.from_dict(all_ds[split])
            post_ds = post_ds.map(process)
            if args.push_to_hub:
                post_ds.select(range(len(post_ds) // 2)).push_to_hub(args.repo_id, split=f"{split}_sft")
                post_ds.select(range(len(post_ds) // 2, len(post_ds))).push_to_hub(args.repo_id, split=f"{split}_prefs")
                if "/" not in args.repo_id:  # find the current user
                    repo_id = f"{api.whoami()['name']}/{args.repo_id}"
                for file, name in zip([__file__, args.constitution_path], ["create_dataset.py", "constitution.json"]):
                    api.upload_file(
                        path_or_fileobj=file,
                        path_in_repo=name,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )

    asyncio.run(main())
