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

import os
import numpy as np
from ruamel import yaml
from tensorboardX import SummaryWriter
from matplotlib import cm
from scipy.io.wavfile import write
import librosa
from tqdm import tqdm
import argparse
from pprint import pprint

import paddle
import paddle.distributed as dist
from parakeet.models.tacotron2.tacotron2 import Tacotron2

from parakeet.utils import io
from parakeet.g2p.en import text_to_sequence
from parakeet.models.waveflow import WaveFlowModule
from parakeet.modules.weight_norm import WeightNormWrapper


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument(
        "--stop_threshold",
        type=float,
        default=0.5,
        help="The threshold of stop token which indicates the time step should stop generate spectrum or not."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1000,
        help="The max length of spectrum when synthesize. If the length of synthetical spectrum is lager than max_len, spectrum will be cut off."
    )

    parser.add_argument(
        "--checkpoint", type=str, help="tacotron2 checkpoint for synthesis")
    parser.add_argument(
        "--vocoder",
        type=str,
        default="griffin-lim",
        choices=['griffin-lim', 'waveflow'],
        help="vocoder method")
    parser.add_argument(
        "--config_vocoder", type=str, help="path of the vocoder config file")
    parser.add_argument(
        "--checkpoint_vocoder",
        type=str,
        help="vocoder checkpoint for synthesis")

    parser.add_argument(
        "--output",
        type=str,
        default="synthesis",
        help="path to save experiment results")


@paddle.fluid.dygraph.no_grad
def synthesis(text_input, args):
    local_rank = dist.get_rank()

    place = 'gpu:{}'.format(local_rank) if args.use_gpu else 'cpu'
    paddle.set_device(place)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # tensorboard
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    writer = SummaryWriter(os.path.join(args.output, 'log'))

    with paddle.utils.unique_name.guard():
        cfg_net = cfg['network']
        model = Tacotron2(
            cfg['audio']['num_mels'], cfg_net['n_frames_per_step'],
            cfg_net['symbols_embedding_dim'], cfg_net['encoder_n_convs'],
            cfg_net['encoder_embedding_dim'], cfg_net['encoder_kernel_size'],
            cfg_net['prenet_dim'], cfg_net['attention_rnn_dim'],
            cfg_net['decoder_rnn_dim'], cfg_net['attention_dim'],
            cfg_net['attention_location_n_filters'],
            cfg_net['attention_location_kernel_size'],
            cfg_net['p_attention_dropout'], cfg_net['p_decoder_dropout'],
            cfg_net['postnet_embedding_dim'], cfg_net['postnet_kernel_size'],
            cfg_net['postnet_n_convs'])
        # Load parameters.
        global_step = io.load_parameters(
            model=model, checkpoint_path=args.checkpoint)
        model.eval()

    # init input
    texts = np.asarray(text_to_sequence(text_input))
    texts = paddle.unsqueeze(paddle.to_tensor(texts, dtype='int64'), [0])

    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(
        texts,
        gate_threshold=args.stop_threshold,
        max_decoder_steps=args.max_len)

    x = np.uint8(cm.viridis(alignments[0].numpy()) * 255)
    writer.add_image('Attention_%d_0' % 0, x, 0, dataformats="HWC")

    if args.vocoder == 'griffin-lim':
        #synthesis use griffin-lim
        wav = synthesis_with_griffinlim(mel_outputs_postnet, cfg['audio'])
    elif args.vocoder == 'waveflow':
        # synthesis use waveflow
        wav = synthesis_with_waveflow(mel_outputs_postnet, args,
                                      args.checkpoint_vocoder)
    else:
        print(
            'vocoder error, we only support griffinlim and waveflow, but recevied %s.'
            % args.vocoder)

    writer.add_audio(text_input + '(' + args.vocoder + ')', wav, 0,
                     cfg['audio']['sr'])
    if not os.path.exists(os.path.join(args.output, 'samples')):
        os.mkdir(os.path.join(args.output, 'samples'))
    write(
        os.path.join(
            os.path.join(args.output, 'samples'), args.vocoder + '.wav'),
        cfg['audio']['sr'], wav)
    print("Synthesis completed !!!")
    writer.close()


def synthesis_with_griffinlim(mel_output, cfg):
    # synthesis with griffin-lim
    mel_output = paddle.transpose(paddle.squeeze(mel_output, [0]), [1, 0])
    mel_output = np.exp(mel_output.numpy())
    basis = librosa.filters.mel(cfg['sr'],
                                cfg['n_fft'],
                                cfg['num_mels'],
                                fmin=cfg['mel_fmin'],
                                fmax=cfg['mel_fmax'])
    inv_basis = np.linalg.pinv(basis)
    spec = np.maximum(1e-10, np.dot(inv_basis, mel_output))

    wav = librosa.griffinlim(
        spec**cfg['power'],
        hop_length=cfg['hop_length'],
        win_length=cfg['win_length'])

    return wav


def synthesis_with_waveflow(mel_output, args, checkpoint):
    args.config = args.config_vocoder
    args.use_fp16 = False
    config = io.add_yaml_config_to_args(args)

    mel_spectrogram = paddle.transpose(paddle.squeeze(mel_output, [0]), [1, 0])
    mel_spectrogram = paddle.unsqueeze(mel_spectrogram, [0])

    # Build model.
    waveflow = WaveFlowModule(config)
    model_dict = paddle.load(checkpoint + ".pdparams")
    waveflow.set_state_dict(model_dict)

    for layer in waveflow.sublayers():
        if isinstance(layer, WeightNormWrapper):
            layer.remove_weight_norm()

    # Run model inference.
    wav = waveflow.synthesize(mel_spectrogram, sigma=config.sigma)
    return wav.numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesis model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(vars(args))
    synthesis(
        "Life was like a box of chocolates, you never know what you're gonna get.",
        args)
