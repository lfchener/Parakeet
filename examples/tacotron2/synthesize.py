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

import argparse
from pathlib import Path
import numpy as np

import paddle
import parakeet
from parakeet.frontend import EnglishCharacter
from parakeet.models.tacotron2 import Tacotron2

from config import get_cfg_defaults


def main(config, args):
    paddle.set_device(args.device)

    # model
    frontend = EnglishCharacter()
    model = Tacotron2.from_pretrained(frontend, config, args.checkpoint_path)
    model.eval()

    # inputs
    input_path = Path(args.input).expanduser()
    with open(input_path, "rt") as f:
        sentences = f.readlines()

    if args.output is None:
        output_dir = input_path.parent / "synthesis"
    else:
        output_dir = Path(args.output).expanduser()
    output_dir.mkdir(exist_ok=True)

    speaker_embedding = None
    if config.model.speaker_embeding and args.speaker_embedding_path is not None:
        speaker_embedding = paddle.unsqueeze(
            paddle.to_tensor(np.load(args.speaker_embedding_path)), 0)

    for i, sentence in enumerate(sentences):
        mel_output, _ = model.predict(
            sentence, speaker_embedding=speaker_embedding)
        mel_output = mel_output.T

        np.save(str(output_dir / f"sentence_{i}"), mel_output)
        if args.verbose:
            print("spectrogram saved at {}".format(output_dir /
                                                   f"sentence_{i}.npy"))


if __name__ == "__main__":
    config = get_cfg_defaults()

    parser = argparse.ArgumentParser(
        description="generate mel spectrogram with TransformerTTS.")
    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="extra config to overwrite the default config")
    parser.add_argument(
        "--checkpoint_path", type=str, help="path of the checkpoint to load.")
    parser.add_argument(
        "--speaker_embedding_path",
        type=str,
        default=None,
        help="path of the speaker embedding file.")
    parser.add_argument("--input", type=str, help="path of the text sentences")
    parser.add_argument("--output", type=str, help="path to save outputs")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device type to use.")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print msg")

    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
