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

import torch

from alignment import DataArguments, ModelArguments, get_peft_config, get_quantization_config, get_tokenizer
from alignment.data import DEFAULT_CHAT_TEMPLATE


class GetQuantizationConfigTest(unittest.TestCase):
    def test_4bit(self):
        model_args = ModelArguments(load_in_4bit=True)
        quantization_config = get_quantization_config(model_args)
        self.assertTrue(quantization_config.load_in_4bit)
        self.assertEqual(quantization_config.bnb_4bit_compute_dtype, torch.float16)
        self.assertEqual(quantization_config.bnb_4bit_quant_type, "nf4")
        self.assertFalse(quantization_config.bnb_4bit_use_double_quant)

    def test_8bit(self):
        model_args = ModelArguments(load_in_8bit=True)
        quantization_config = get_quantization_config(model_args)
        self.assertTrue(quantization_config.load_in_8bit)

    def test_no_quantization(self):
        model_args = ModelArguments()
        quantization_config = get_quantization_config(model_args)
        self.assertIsNone(quantization_config)


class GetTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_args = ModelArguments(model_name_or_path="HuggingFaceH4/zephyr-7b-alpha")

    def test_right_truncation_side(self):
        tokenizer = get_tokenizer(self.model_args, DataArguments(truncation_side="right"))
        self.assertEqual(tokenizer.truncation_side, "right")

    def test_left_truncation_side(self):
        tokenizer = get_tokenizer(self.model_args, DataArguments(truncation_side="left"))
        self.assertEqual(tokenizer.truncation_side, "left")

    def test_default_chat_template(self):
        tokenizer = get_tokenizer(self.model_args, DataArguments())
        self.assertEqual(tokenizer.chat_template, DEFAULT_CHAT_TEMPLATE)

    def test_chatml_chat_template(self):
        chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer = get_tokenizer(self.model_args, DataArguments(chat_template=chat_template))
        self.assertEqual(tokenizer.chat_template, chat_template)


class GetPeftConfigTest(unittest.TestCase):
    def test_peft_config(self):
        model_args = ModelArguments(use_peft=True, lora_r=42, lora_alpha=0.66, lora_dropout=0.99)
        peft_config = get_peft_config(model_args)
        self.assertEqual(peft_config.r, 42)
        self.assertEqual(peft_config.lora_alpha, 0.66)
        self.assertEqual(peft_config.lora_dropout, 0.99)

    def test_no_peft_config(self):
        model_args = ModelArguments(use_peft=False)
        peft_config = get_peft_config(model_args)
        self.assertIsNone(peft_config)
