# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from unittest import TestCase

from datasets import Dataset
from transformers import AutoTokenizer

from alignment import apply_chat_template, decontaminate_humaneval


class DecontamintateHumanEvalTest(TestCase):
    """Test we decontaminate HumanEval samples correctly"""

    def setUp(self) -> None:
        # Create a dataset with a HumanEval sample wrapped in some fake text
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"content": "Hello", "role": "user"}],
                    [
                        {
                            "content": 'Hello, I am\nfrom\n\n typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
                            "role": "assistant",
                        }
                    ],
                ]
            }
        )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        self.dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})

    def test_decontamination(self):
        """Test we decontaminate HumanEval samples correctly"""
        decontaminated_dataset = self.dataset.filter(decontaminate_humaneval, batched=True)
        # Check we recover just the first message
        self.assertEqual(decontaminated_dataset[0]["text"], self.dataset[0]["text"])
