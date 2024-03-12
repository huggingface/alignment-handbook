# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List

from datasets import load_dataset


# HumanEval solutions that are considered simple/generic enough to be kept in the training dataset
HUMAN_EVAL_STRINGS_OK = ["return x + y", "return len(string)", "return n**2", "return " ".join(strings)"]


def extract_docstring(prompt: str) -> str:
    if '"""' in prompt:
        if prompt.count('"""') == 2:
            return prompt.split('"""')[1].strip()
        elif prompt.count('"""') == 4:
            return prompt.split('"""')[3].strip()
        else:
            raise ValueError()
    elif "'''" in prompt:
        assert prompt.count("'''") == 2
        return prompt.split("'''")[1].strip()
    else:
        raise ValueError()


def human_eval_docstrings() -> List[str]:
    ds = load_dataset("openai_humaneval", split="test")
    docstrings = [extract_docstring(v["prompt"]) for v in ds]
    return docstrings


def load_dataset_column(dataset: str, column: str, split: str, name=None) -> List[str]:
    ds = load_dataset(dataset, split=split, name=name)
    res = [sample[column].strip() for sample in ds]
    # Only return non-empty strings
    return [sample for sample in res if len(sample) > 0]


FILTER_OUT = {
    "human_eval_docstrings": human_eval_docstrings(),
    "human_eval_solutions": [
        s
        for s in load_dataset_column("openai_humaneval", "canonical_solution", "test")
        if s not in HUMAN_EVAL_STRINGS_OK
    ],
}


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def decontaminate_humaneval(
    samples: List[Dict[str, Any]], text_column: str = "text", filter_out: Dict[str, List[str]] = FILTER_OUT
) -> List[Dict[str, Any]]:
    """
    filter_out: Dict[str, List[str]] mapping from benchmark name to list of strings that need to be
    filtered-out.
    Return a list where each element is True if the corresponding file should be included in the dataset.
    Otherwise, the element is False.
    """
    output = []

    for content in samples[text_column]:
        content = normalize_whitespace(content.lower())
        matched = False
        for _, substrings in filter_out.items():
            for substring in substrings:
                if normalize_whitespace(substring.lower()) in content:
                    matched = True
                    break
            if matched:
                break
        # we keep files that are not matched
        output.append(not matched)

    return output
