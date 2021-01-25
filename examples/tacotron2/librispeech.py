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
from parakeet.data.batch import batch_spec, batch_text_id


class LibriSpeech(Dataset):
    """A simple dataset adaptor for the processed librispeech dataset."""

    def __init__(self, root):
        # Load metadata
        mel_dir = os.path.join(os.path.dirname(root), "mels")
        embed_dir = os.path.join(os.path.dirname(root), "embeds")
        with open(
                os.path.join(os.path.dirname(root), "train.txt"),
                encoding="utf-8") as f:
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


class LibriSpeechCollector(object):
    """A simple callable to batch LibriSpeech examples."""

    def __init__(self, padding_idx=0, padding_value=0.,
                 padding_stop_token=1.0):
        self.padding_idx = padding_idx
        self.padding_value = padding_value
        self.padding_stop_token = padding_stop_token

    def __call__(self, examples):
        texts = []
        mels = []
        embeds = []
        text_lens = []
        mel_lens = []
        stop_tokens = []
        for data in examples:
            text, mel, embed = data
            text = np.array(text, dtype=np.int64)
            text_lens.append(len(text))
            mels.append(mel)
            texts.append(text)
            embeds.append(embed)
            mel_lens.append(mel.shape[0])
            stop_token = np.zeros([mel.shape[0] - 1], dtype=np.float32)
            stop_tokens.append(np.append(stop_token, 1.0))

        # Sort by text_len in descending order
        texts = [
            i for i, _ in sorted(
                zip(texts, text_lens), key=lambda x: x[1], reverse=True)
        ]
        mels = [
            i for i, _ in sorted(
                zip(mels, text_lens), key=lambda x: x[1], reverse=True)
        ]

        embeds = [
            i for i, _ in sorted(
                zip(embeds, text_lens), key=lambda x: x[1], reverse=True)
        ]

        mel_lens = [
            i for i, _ in sorted(
                zip(mel_lens, text_lens), key=lambda x: x[1], reverse=True)
        ]

        stop_tokens = [
            i for i, _ in sorted(
                zip(stop_tokens, text_lens), key=lambda x: x[1], reverse=True)
        ]

        text_lens = sorted(text_lens, reverse=True)

        # Pad sequence with largest len of the batch
        texts = batch_text_id(texts, pad_id=self.padding_idx)
        mels = batch_spec(mels, pad_value=self.padding_value, time_major=True)
        stop_tokens = batch_text_id(
            stop_tokens, pad_id=self.padding_stop_token, dtype=mels[0].dtype)

        return (texts, mels, text_lens, mel_lens, stop_tokens, embeds)
