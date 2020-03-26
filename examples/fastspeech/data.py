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

from paddle import fluid
from parakeet import g2p
from parakeet import audio
from parakeet.data.sampler import *
from parakeet.data.datacargo import DataCargo
from parakeet.data.batch import TextIDBatcher, SpecBatcher
from parakeet.data.dataset import DatasetMixin, TransformDataset, CacheDataset
from parakeet.models.transformer_tts.utils import *


class LJSpeechLoader:
    def __init__(self,
                 config,
                 args,
                 nranks,
                 rank,
                 is_vocoder=False,
                 shuffle=True):
        place = fluid.CUDAPlace(rank) if args.use_gpu else fluid.CPUPlace()

        LJSPEECH_ROOT = Path(args.data_path)
        metadata = LJSpeechMetaData(LJSPEECH_ROOT)
        transformer = LJSpeech(config)
        dataset = TransformDataset(metadata, transformer)
        dataset = CacheDataset(dataset)

        sampler = DistributedSampler(
            len(metadata), nranks, rank, shuffle=shuffle)

        assert args.batch_size % nranks == 0
        each_bs = args.batch_size // nranks
        if is_vocoder:
            dataloader = DataCargo(
                dataset,
                sampler=sampler,
                batch_size=each_bs,
                shuffle=shuffle,
                batch_fn=batch_examples_vocoder,
                drop_last=True)
        else:
            dataloader = DataCargo(
                dataset,
                sampler=sampler,
                batch_size=each_bs,
                shuffle=shuffle,
                batch_fn=batch_examples,
                drop_last=True)

        self.reader = fluid.io.DataLoader.from_generator(
            capacity=32,
            iterable=True,
            use_double_buffer=True,
            return_list=True)
        self.reader.set_batch_generator(dataloader, place)


class LJSpeechMetaData(DatasetMixin):
    def __init__(self, root):
        self.root = Path(root)
        self._wav_dir = "train.txt"
        with open(self._wav_dir, "r", encoding="utf-8") as f:
            self.text = []
            for line in f.readlines():
                self.text.append(line)


    def get_example(self, i):
        mel_gt_name = os.path.join(
            "mels", "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        D = np.load(os.path.join("./alignments", str(i)+".npy"))

        character = self.text[i][0:len(self.text[i])-1]

        return character, mel_gt_target, D

    def __len__(self):
        return len(self.text)


class LJSpeech(object):
    def __init__(self, config):
        super(LJSpeech, self).__init__()
        self.config = config
        self._ljspeech_processor = audio.AudioProcessor(
            sample_rate=config['audio']['sr'],
            num_mels=config['audio']['num_mels'],
            min_level_db=config['audio']['min_level_db'],
            ref_level_db=config['audio']['ref_level_db'],
            n_fft=config['audio']['n_fft'],
            win_length=config['audio']['win_length'],
            hop_length=config['audio']['hop_length'],
            power=config['audio']['power'],
            preemphasis=config['audio']['preemphasis'],
            signal_norm=True,
            symmetric_norm=False,
            max_norm=1.,
            mel_fmin=0,
            mel_fmax=None,
            clip_norm=True,
            griffin_lim_iters=60,
            do_trim_silence=False,
            sound_norm=False)

    def __call__(self, metadatum):
        """All the code for generating an Example from a metadatum. If you want a 
        different preprocessing pipeline, you can override this method. 
        This method may require several processor, each of which has a lot of options.
        In this case, you'd better pass a composed transform and pass it to the init
        method.
        """
        character, mel_gt_target, D = metadatum

        # load -> trim -> preemphasis -> stft -> magnitude -> mel_scale -> logscale -> normalize
        D = D.astype(np.float32)
        mel_gt_target = mel_gt_target.astype(np.float32)
        phonemes = np.array(
            g2p.en.text_to_sequence(character), dtype=np.int64)
        return (phonemes, mel_gt_target, D)  # maybe we need to implement it as a map in the future


def batch_examples(batch):
    texts = []
    mels = []
    mel_inputs = []
    mel_lens = []
    text_lens = []
    pos_texts = []
    pos_mels = []
    Ds = []
    for data in batch:
        text, mel, D = data
        mel_inputs.append(
            np.concatenate(
                [np.zeros([mel.shape[0], 1], np.float32), mel[:, :-1]],
                axis=-1))
        mel_lens.append(mel.shape[0])
        text_lens.append(len(text))
        pos_texts.append(np.arange(1, len(text) + 1))
        pos_mels.append(np.arange(1, mel.shape[0] + 1))
        mels.append(mel)
        texts.append(text)
        Ds.append(D)

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
    mel_inputs = [
        i
        for i, _ in sorted(
            zip(mel_inputs, text_lens), key=lambda x: x[1], reverse=True)
    ]
    mel_lens = [
        i
        for i, _ in sorted(
            zip(mel_lens, text_lens), key=lambda x: x[1], reverse=True)
    ]
    pos_texts = [
        i
        for i, _ in sorted(
            zip(pos_texts, text_lens), key=lambda x: x[1], reverse=True)
    ]
    pos_mels = [
        i
        for i, _ in sorted(
            zip(pos_mels, text_lens), key=lambda x: x[1], reverse=True)
    ]
    Ds = [
        i
        for i, _ in sorted(
            zip(Ds, text_lens), key=lambda x: x[1], reverse=True)
    ]
    text_lens = sorted(text_lens, reverse=True)

    # Pad sequence with largest len of the batch
    texts = TextIDBatcher(pad_id=0)(texts)  #(B, T)
    pos_texts = TextIDBatcher(pad_id=0)(pos_texts)  #(B,T)
    Ds = TextIDBatcher(pad_id=0)(Ds).astype(np.float32)
    pos_mels = TextIDBatcher(pad_id=0)(pos_mels)  #(B,T)
    mels = SpecBatcher(pad_value=0.)(mels)  #(B,T,num_mels)
    mel_inputs = SpecBatcher(pad_value=0.)(mel_inputs)  #(B,T,num_mels)
    enc_slf_mask = get_attn_key_pad_mask(pos_texts, texts).astype(np.float32)
    enc_query_mask = get_non_pad_mask(pos_texts).astype(np.float32)
    dec_slf_mask = get_dec_attn_key_pad_mask(pos_mels,
                                             mel_inputs).astype(np.float32)
    enc_dec_mask = get_attn_key_pad_mask(enc_query_mask[:, :, 0],
                                         mel_inputs).astype(np.float32)
    dec_query_slf_mask = get_non_pad_mask(pos_mels).astype(np.float32)
    dec_query_mask = get_non_pad_mask(pos_mels).astype(np.float32)

    return (texts, mels, mel_inputs, pos_texts, pos_mels, np.array(text_lens),
            np.array(mel_lens), enc_slf_mask, enc_query_mask, dec_slf_mask,
            enc_dec_mask, dec_query_slf_mask, dec_query_mask, Ds)
