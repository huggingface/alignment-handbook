# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest
from copy import deepcopy

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from alignment import DataArguments, ModelArguments, apply_chat_template, get_datasets, get_tokenizer
from alignment.data import maybe_insert_system_message


class GetDatasetsTest(unittest.TestCase):
    """Each of these test datasets has 100 examples"""

    def test_loading_data_args(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.5,
            "HuggingFaceH4/testing_self_instruct_small": 0.3,
            "HuggingFaceH4/testing_codealpaca_small": 0.2,
        }
        data_args = DataArguments(dataset_mixer=dataset_mixer)
        datasets = get_datasets(data_args)
        self.assertEqual(len(datasets["train"]), 100)
        self.assertEqual(len(datasets["test"]), 300)

    def test_loading_data_dict(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.5,
            "HuggingFaceH4/testing_self_instruct_small": 0.3,
            "HuggingFaceH4/testing_codealpaca_small": 0.2,
        }
        datasets = get_datasets(dataset_mixer)
        self.assertEqual(len(datasets["train"]), 100)
        self.assertEqual(len(datasets["test"]), 300)

    def test_loading_with_unit_fractions(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 1.0,
            "HuggingFaceH4/testing_self_instruct_small": 1.0,
            "HuggingFaceH4/testing_codealpaca_small": 1.0,
        }
        datasets = get_datasets(dataset_mixer)
        self.assertEqual(len(datasets["train"]), 300)
        self.assertEqual(len(datasets["test"]), 300)

    def test_loading_with_fractions_greater_than_unity(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.7,
            "HuggingFaceH4/testing_self_instruct_small": 0.4,
        }
        datasets = get_datasets(dataset_mixer)
        self.assertEqual(len(datasets["train"]), 70 + 40)
        self.assertEqual(len(datasets["test"]), 200)

    def test_loading_fails_with_negative_fractions(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.7,
            "HuggingFaceH4/testing_self_instruct_small": -0.3,
        }
        with pytest.raises(ValueError, match=r"Dataset fractions cannot be negative."):
            get_datasets(dataset_mixer)

    def test_loading_single_split_with_unit_fractions(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 1.0,
        }
        datasets = get_datasets(dataset_mixer, splits=["test"])
        self.assertEqual(len(datasets["test"]), 100)
        self.assertRaises(KeyError, lambda: datasets["train"])


class ApplyChatTemplateTest(unittest.TestCase):
    def setUp(self):
        model_args = ModelArguments(model_name_or_path="HuggingFaceH4/zephyr-7b-alpha")
        data_args = DataArguments()
        self.tokenizer = get_tokenizer(model_args, data_args)
        self.dataset = Dataset.from_dict(
            {
                "prompt": ["Hello!"],
                "messages": [
                    [
                        {"role": "system", "content": "You are a happy chatbot"},
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Bonjour!"},
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I am doing well, thanks!"},
                    ]
                ],
                "chosen": [
                    [
                        {"role": "system", "content": "You are a happy chatbot"},
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Bonjour!"},
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I am doing well, thanks!"},
                    ]
                ],
                "rejected": [
                    [
                        {"role": "system", "content": "You are a happy chatbot"},
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Bonjour!"},
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "Not so good tbh"},
                    ]
                ],
            }
        )

    def test_maybe_insert_system_message(self):
        # does not accept system prompt
        mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        # accepts system prompt. use codellama since it has no HF token reqiurement
        llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        messages_sys_excl = [{"role": "user", "content": "Tell me a joke."}]
        messages_sys_incl = [{"role": "system", "content": ""}, {"role": "user", "content": "Tell me a joke."}]

        mistral_messages = deepcopy(messages_sys_excl)
        llama_messages = deepcopy(messages_sys_excl)
        maybe_insert_system_message(mistral_messages, mistral_tokenizer)
        maybe_insert_system_message(llama_messages, llama_tokenizer)

        # output from mistral should not have a system message, output from llama should
        self.assertEqual(mistral_messages, messages_sys_excl)
        self.assertEqual(llama_messages, messages_sys_incl)

    def test_sft(self):
        dataset = self.dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer, "task": "sft"},
            remove_columns=self.dataset.column_names,
        )
        self.assertDictEqual(
            dataset[0],
            {
                "text": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\nI am doing well, thanks!</s>\n"
            },
        )

    def test_generation(self):
        # Remove last turn from messages
        dataset = self.dataset.map(lambda x: {"messages": x["messages"][:-1]})
        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer, "task": "generation"},
            remove_columns=self.dataset.column_names,
        )
        self.assertDictEqual(
            dataset[0],
            {
                "text": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\n"
            },
        )

    def test_rm(self):
        dataset = self.dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer, "task": "rm"},
            remove_columns=self.dataset.column_names,
        )
        self.assertDictEqual(
            dataset[0],
            {
                "text_chosen": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\nI am doing well, thanks!</s>\n",
                "text_rejected": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n<|assistant|>\nNot so good tbh</s>\n",
            },
        )

    def test_dpo(self):
        dataset = self.dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer, "task": "dpo"},
            remove_columns=self.dataset.column_names,
        )
        self.assertDictEqual(
            dataset[0],
            {
                "text_prompt": "<|system|>\nYou are a happy chatbot</s>\n<|user|>\nHello!</s>\n<|assistant|>\nBonjour!</s>\n<|user|>\nHow are you?</s>\n",
                "text_chosen": "<|assistant|>\nI am doing well, thanks!</s>\n",
                "text_rejected": "<|assistant|>\nNot so good tbh</s>\n",
            },
        )
