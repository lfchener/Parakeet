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

import numpy as np
import argparse
import os
import time
import math
from pathlib import Path
from pprint import pprint
from ruamel import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from collections import OrderedDict
from tensorboardX import SummaryWriter
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import paddle
from parakeet.models.tacotron2.tacotron2 import Tacotron2
from data import LJSpeechLoader
from parakeet.utils import io


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument("--data", type=str, help="path of LJspeech dataset")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")

    parser.add_argument(
        "--output",
        type=str,
        default="experiment",
        help="path to save experiment results")


def main(args):
    local_rank = dg.parallel.Env().local_rank
    nranks = dg.parallel.Env().nranks
    parallel = nranks > 1

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    global_step = 0
    place = fluid.CUDAPlace(dg.parallel.Env()
                            .dev_id) if args.use_gpu else fluid.CPUPlace()
    fluid.enable_dygraph(place)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    writer = SummaryWriter(os.path.join(args.output,
                                        'log')) if local_rank == 0 else None

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
    learning_rate = cfg['train']['learning_rate']
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=learning_rate,
        parameter_list=model.parameters(),
        regularization=fluid.regularizer.L2Decay(cfg['train']['weight_decay']),
        grad_clip=fluid.clip.GradientClipByGlobalNorm(cfg['train'][
            'grad_clip_thresh']))

    model.train()
    # Load parameters.
    global_step = io.load_parameters(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=os.path.join(args.output, 'checkpoints'),
        iteration=args.iteration,
        checkpoint_path=args.checkpoint)
    print("Rank {}: checkpoint loaded.".format(local_rank))

    if parallel:
        strategy = dg.parallel.prepare_context()
        model = fluid.dygraph.parallel.DataParallel(model, strategy)

    reader = LJSpeechLoader(
        cfg['audio'],
        place,
        args.data,
        cfg['train']['batch_size'],
        nranks,
        local_rank,
        shuffle=True).reader
    iterator = iter(tqdm(reader))

    while global_step <= cfg['train']['max_iteration']:
        try:
            batch = next(iterator)
        except StopIteration as e:
            iterator = iter(tqdm(reader))
            batch = next(iterator)

        (texts, mels, text_lens, output_lens, stop_tokens) = batch

        global_step += 1

        #Forward
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(
            texts, text_lens, mels, output_lens)

        mel_loss = paddle.nn.MSELoss()(mel_outputs, mels)
        post_mel_loss = paddle.nn.MSELoss()(mel_outputs_postnet, mels)
        gate_loss = paddle.nn.BCELoss()(gate_outputs, stop_tokens)
        total_loss = mel_loss + post_mel_loss + gate_loss

        if local_rank == 0:
            writer.add_scalar('mel_loss', mel_loss.numpy(), global_step)
            writer.add_scalar('post_mel_loss',
                              post_mel_loss.numpy(), global_step)
            writer.add_scalar('gate_loss', gate_loss.numpy(), global_step)
            writer.add_scalar('learning_rate', optimizer._learning_rate,
                              global_step)
            if global_step % cfg['train']['image_interval'] == 1:
                idx = np.random.randint(0, alignments.shape[0] - 1)
                x = np.uint8(cm.viridis(alignments[idx].numpy()) * 255)
                writer.add_image(
                    'Attention_%d_0' % global_step, x, 0, dataformats="HWC")

                fig, ax = plt.subplots(figsize=(12, 3))
                ax.scatter(
                    range(len(stop_tokens[idx].numpy())),
                    stop_tokens[idx].numpy(),
                    alpha=0.5,
                    color='green',
                    marker='+',
                    s=1,
                    label='target')
                ax.scatter(
                    range(len(gate_outputs[idx].numpy())),
                    gate_outputs[idx].numpy(),
                    alpha=0.5,
                    color='red',
                    marker='.',
                    s=1,
                    label='predicted')

                plt.xlabel("Frames (Green target, Red predicted)")
                plt.ylabel("Gate State")
                plt.tight_layout()

                fig.canvas.draw()

                writer.add_figure('stop_token_%d' % global_step, fig,
                                  global_step)

        if parallel:
            total_loss = model.scale_loss(total_loss)
            total_loss.backward()
            model.apply_collective_grads()
        else:
            total_loss.backward()
        optimizer.minimize(total_loss)
        model.clear_gradients()

        # save checkpoint
        if local_rank == 0 and global_step % cfg['train'][
                'checkpoint_interval'] == 0:
            io.save_parameters(
                os.path.join(args.output, 'checkpoints'), global_step, model,
                optimizer)

    if local_rank == 0:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Tacotron model.")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(vars(args))
    main(args)
