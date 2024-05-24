"""This Python script loops over a directory with checkpoints generated by the
`transformers.Trainer` or any other subclass of it, and uploads those to the Hugging
Face Hub, while https://github.com/huggingface/transformers/issues/30724 is fixed.

Note that this script requires you to either set `HF_TOKEN` or call `huggingface-cli login`
in advance to work. Also note that each checkpoint will be uploaded only if matching the
default structure i.e. under a directory named `checkpoint-...` under the provided directory
and only the JSON and `safetensors` file will be uploaded. When uploading one checkpoint, the
`epoch-...` suffix will be added to the `repo_id` provided, in the order the checkpoints are
sorted in the directory i.e. assuming `save_strategy="epoch"`.

Args:
    ckpt-dir: The directory containing the checkpoints.
    repo-id: The repository ID to use for the checkpoints. Note that the `epoch-...` suffix
        will be added to this ID.
    private: Whether the repository should be private or not.
    include-last: Whether to include the last checkpoint or not. Default is False, since the
        last checkpoint is usually uploaded by default when the training is finished if the
        `push_to_hub` argument is set to True within the `transformers.Trainer`.

Usage:
    >>> python save_ckpt.py --ckpt-dir data/... --repo-id ... --private

Example:
    >>> python save_ckpt.py --ckpt-dir data/mistral-7b-v0.1/ --repo-id mistral-7b-v0.1-sft --private
"""
import argparse
import os

from huggingface_hub import HfApi


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # 0. Instantiate the HfApi
    api = HfApi()

    # 1. Loop over the provided directory and keep all the directories
    # that start with `checkpoint-` in order
    ckpt_dirs = []
    for ckpt_dir in os.listdir(args.ckpt_dir):
        if ckpt_dir.startswith("checkpoint-"):
            ckpt_dirs.append(ckpt_dir)
    if not args.include_last:
        ckpt_dirs = ckpt_dirs[:-1]

    print(
        f"Found {len(ckpt_dirs)} checkpoints to upload"
        f"{' excluding the last one' if not args.include_last else ''}"
    )
    print(f"Checkpoints to upload: {ckpt_dirs}")

    # 2. Loop over each checkpoint directory keeping an epoch count
    for epoch, ckpt_dir in enumerate(ckpt_dirs, 1):
        repo_id = f"{args.repo_id}-epoch-{epoch}"

        # 3. Create the repository in the Hugging Face Hub
        print(f"Creating repository {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            private=args.private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"Repository {repo_id} created")

        folder_path = os.path.join(args.ckpt_dir, ckpt_dir)

        # 4. Upload the folder to the repository, but only the JSON and `safetensors` files
        print(f"Uploading JSON files from {ckpt_dir} to repository {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder_path,
            path_in_repo=".",
            commit_message="Uploading JSON files from checkpoint",
            allow_patterns="*.json",
        )

        print(f"Uploading `safetensors` files from {ckpt_dir} to repository {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder_path,
            path_in_repo=".",
            commit_message="Uploading `safetensors`files from checkpoint",
            allow_patterns="*safetensors*",
        )

        print(f"Checkpoint {ckpt_dir} uploaded to repository {repo_id}")