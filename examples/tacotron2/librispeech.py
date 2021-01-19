# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from pathlib import Path
import pickle
import numpy as np
from paddle.io import Dataset, DataLoader
from parakeet.frontend import EnglishCharacter


class LJSpeech(Dataset):
    """A simple dataset adaptor for the processed librispeech dataset."""

    def __init__(self, root):
        # Load metadata
        mel_dir = os.path.join(os.path.dirname(root), "mels")
        embed_dir = os.path.join(os.path.dirname(root), "embeds")
        with open(root, 'rb', encoding="utf-8") as f:
            metadata = [line.strip().split("|") for line in f]
        frontend = EnglishCharacter()

        records = []
        for _, mel_name, embed_name, _, _, text in metadata:
            ids = frontend(text)
            mel_name = os.path.join(mel_dir, mel_name)
            embed_name = os.path.join(embed_dir, embed_name)
            records.append((ids, mel_name, embed_name))
        self.records = records

    def __getitem__(self, i):
        ids, mel_name, embed_name = self.records[i]
        mel = np.load(mel_name)
        embed = np.load(embed_name)
        return ids, mel, embed

    def __len__(self):
        return len(self.records)
