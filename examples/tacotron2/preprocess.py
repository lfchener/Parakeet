from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import librosa
import csv
import pickle
from tqdm import tqdm
from ruamel import yaml

from paddle.io import Dataset
from parakeet import g2p
from parakeet import audio


class LJSpeechMetaData(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self._wav_dir = self.root.joinpath("wavs")
        csv_path = self.root.joinpath("metadata.csv")
        self._table = pd.read_csv(
            csv_path,
            sep="|",
            header=None,
            quoting=csv.QUOTE_NONE,
            names=["fname", "raw_text", "normalized_text"])

    def __getitem__(self, idx):
        fname, raw_text, normalized_text = self._table.iloc[idx]
        fpath = str(self._wav_dir.joinpath(fname + ".wav"))
        return fpath, normalized_text, fname

    def __len__(self):
        return len(self._table)


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
        fpath, normalized_text, fname = metadatum

        wav, _ = librosa.load(str(fpath))
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
        return (mel, character, fname)

class TransformDataset(Dataset):

    def __init__(self, dataset, transform):
        """Dataset which is transformed from another with a transform.
        Args:
            dataset (Dataset): the base dataset.
            transform (callable): the transform which takes an example of the base dataset as parameter and return a new example.
        """
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        in_data = self._dataset[i]
        return self._transform(in_data)

def prepare_dataset(config, data_path):
    LJSPEECH_ROOT = Path(data_path)
    metadata = LJSpeechMetaData(LJSPEECH_ROOT)
    transformer = LJSpeech(
        sr=config['sr'],
        n_fft=config['n_fft'],
        num_mels=config['num_mels'],
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        mel_fmin=config['mel_fmin'],
        mel_fmax=config['mel_fmax'])
    dataset = TransformDataset(metadata, transformer)
    data_dir = LJSPEECH_ROOT.joinpath("prepared_data")
    chars_dict = dict()
    pbar = tqdm(dataset)
    for d in pbar:
        mel, character, fname = d
        np.save(data_dir.joinpath(fname), mel)
        chars_dict[fname] = character
    with open(data_dir.joinpath('characters.pkl'), 'wb') as f:
        pickle.dump(chars_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare dataset.")
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--data", type=str, help="path of LJspeech dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    prepare_dataset(config['audio'], args.data)



