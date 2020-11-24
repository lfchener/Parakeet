# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import csv
import pickle

from paddle import fluid
from paddle.io import Dataset, DistributedBatchSampler, DataLoader
from parakeet import g2p
from parakeet import audio
from parakeet.data.sampler import *
from parakeet.data.datacargo import DataCargo
from parakeet.data.batch import TextIDBatcher, SpecBatcher
from parakeet.data.dataset import DatasetMixin, TransformDataset, CacheDataset, SliceDataset
from parakeet.models.transformer_tts.utils import *


class LJSpeechLoader:
    def __init__(self,
                 config,
                 place,
                 data_path,
                 batch_size,
                 nranks,
                 rank,
                 shuffle=True):

        LJSPEECH_ROOT = Path(data_path)
        dataset = LJSpeechDataset(LJSPEECH_ROOT)

        sampler = DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

        self.dataloader = DataLoader(
            dataset,
            places=place,
            batch_sampler=sampler,
            collate_fn=batch_examples,
            num_workers=0,
            use_shared_memory=True)


class LJSpeechDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.data_dir = self.root.joinpath("prepared_data")
        chars_path = self.data_dir.joinpath("characters.pkl")
        with open(chars_path, 'rb') as f:
            self.chars_dict = list(pickle.load(f).items())

    def __getitem__(self, idx):
        fname, character = self.chars_dict[idx]
        fname = str(self.data_dir.joinpath(fname + ".npy"))
        mel = np.load(fname)

        return mel, character

    def __len__(self):
        return len(self.chars_dict)


class LJSpeech(object):
    def __init__(self,
                 sr=22050,
                 n_fft=2048,
                 num_mels=80,
                 win_length=1024,
                 hop_length=256,
                 mel_fmin=0,
                 mel_fmax=None):
        super(LJSpeech, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

    def __call__(self, metadatum):
        """All the code for generating an Example from a metadatum. If you want a 
        different preprocessing pipeline, you can override this method. 
        This method may require several processor, each of which has a lot of options.
        In this case, you'd better pass a composed transform and pass it to the init
        method.
        """
        fname, normalized_text = metadatum

        wav, _ = librosa.load(str(fname))
        spec = librosa.stft(
            y=wav,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length)
        mag = np.abs(spec)
        mel = librosa.filters.mel(self.sr,
                                  self.n_fft,
                                  n_mels=self.num_mels,
                                  fmin=self.mel_fmin,
                                  fmax=self.mel_fmax)
        mel = np.matmul(mel, mag)
        mel = np.log(np.maximum(mel, 1e-5))
        character = np.array(
            g2p.en.text_to_sequence(normalized_text), dtype=np.int64)
        return (mel, character)


def batch_examples(batch):
    texts = []
    mels = []
    text_lens = []
    output_lens = []
    stop_tokens = []
    for data in batch:
        mel, text = data
        text_lens.append(len(text))
        mels.append(mel)
        texts.append(text)
        output_lens.append(mel.shape[1])
        stop_token = np.zeros([mel.shape[1] - 1], dtype=np.float32)
        stop_tokens.append(np.append(stop_token, 1.0))

    # Sort by text_len in descending order
    texts = [
        i
        for i, _ in sorted(
            zip(texts, text_lens), key=lambda x: x[1], reverse=True)
    ]
    mels = [
        i
        for i, _ in sorted(
            zip(mels, text_lens), key=lambda x: x[1], reverse=True)
    ]

    output_lens = [
        i
        for i, _ in sorted(
            zip(output_lens, text_lens), key=lambda x: x[1], reverse=True)
    ]

    stop_tokens = [
        i
        for i, _ in sorted(
            zip(stop_tokens, text_lens), key=lambda x: x[1], reverse=True)
    ]

    text_lens = sorted(text_lens, reverse=True)

    # Pad sequence with largest len of the batch
    texts = TextIDBatcher(pad_id=0)(texts)  #(B, T)
    mels = np.transpose(
        SpecBatcher(pad_value=0.)(mels),
        axes=(0, 2, 1)).astype('float64')  #(B,T,C)
    stop_tokens = TextIDBatcher(
        pad_id=1, dtype=np.float32)(stop_tokens).astype('float64')

    return (texts, mels, text_lens, output_lens, stop_tokens)
