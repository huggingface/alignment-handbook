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

import os
import unittest

from alignment import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig


class H4ArgumentParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
        self.yaml_file_path = "tests/fixtures/config_sft_full.yaml"

    def test_load_yaml(self):
        model_args, data_args, training_args = self.parser.parse_yaml_file(os.path.abspath(self.yaml_file_path))
        self.assertEqual(model_args.model_name_or_path, "mistralai/Mistral-7B-v0.1")

    def test_load_yaml_and_args(self):
        command_line_args = [
            "--model_name_or_path=test",
            "--use_peft=true",
            "--lora_r=16",
            "--lora_dropout=0.5",
        ]
        model_args, data_args, training_args = self.parser.parse_yaml_and_args(
            os.path.abspath(self.yaml_file_path), command_line_args
        )
        self.assertEqual(model_args.model_name_or_path, "test")
        self.assertEqual(model_args.use_peft, True)
        self.assertEqual(model_args.lora_r, 16)
        self.assertEqual(model_args.lora_dropout, 0.5)
