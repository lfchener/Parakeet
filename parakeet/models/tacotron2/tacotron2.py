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

from math import sqrt
from parakeet.models.tacotron2.encoder import Encoder
from parakeet.models.tacotron2.decoder import Decoder
from parakeet.models.transformer_tts.post_convnet import PostConvNet
from parakeet.g2p.text.symbols import symbols
import paddle
from paddle import nn
import time


class Tacotron2(nn.Layer):
    def __init__(self,
                 n_mels=80,
                 n_frames_per_step=1,
                 symbols_embedding_dim=512,
                 encoder_n_convs=3,
                 encoder_embedding_dim=512,
                 encoder_kernel_size=5,
                 prenet_dim=256,
                 attention_rnn_dim=1024,
                 decoder_rnn_dim=1024,
                 attention_dim=128,
                 attention_location_n_filters=32,
                 attention_location_kernel_size=31,
                 p_attention_dropout=0.1,
                 p_decoder_dropout=0.1,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_n_convs=5):
        super(Tacotron2, self).__init__()
        self.n_mel_channels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.n_symbols = len(symbols) + 1
        std = sqrt(2.0 / (self.n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding = nn.Embedding(
            #self.n_symbols, symbols_embedding_dim,
            148, symbols_embedding_dim,
            #padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-val, high=val)))
                #trainable=False))
        self.encoder = Encoder(encoder_n_convs, encoder_embedding_dim,
                               encoder_kernel_size)
        self.decoder = Decoder(
            n_mels, n_frames_per_step, encoder_embedding_dim, prenet_dim,
            attention_rnn_dim, decoder_rnn_dim, attention_dim,
            attention_location_n_filters, attention_location_kernel_size,
            p_attention_dropout, p_decoder_dropout)
        self.postnet = PostConvNet(
            n_mels=n_mels,
            num_hidden=postnet_embedding_dim,
            filter_size=postnet_kernel_size,
            padding=int((postnet_kernel_size - 1) / 2),
            num_conv=postnet_n_convs,
            outputs_per_step=1,
            dropout=0.0, #0.5
            batchnorm_last=True)

    def forward(self, text_inputs, text_lens, mels, output_lens):        
        embedded_inputs = self.embedding(text_inputs)  #(B, T, C)
        encoder_outputs = self.encoder(embedded_inputs, text_lens)  #(B, T, C)
        
        #import pdb; pdb.set_trace()
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels,
            memory_lens=text_lens)  #[B, T, C], [B, T], [B, T]

        mel_outputs_postnet = self.postnet(mel_outputs)  #[B, T, C]
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet  #[B, T, C] 

        if output_lens is not None:
            mask = paddle.tensor.unsqueeze(
                paddle.fluid.layers.sequence_mask(x=output_lens), [-1])  #[B, T, 1]
            mel_outputs = mel_outputs * mask  #[B, T, C]
            mel_outputs_postnet = mel_outputs_postnet * mask  #[B, T, C]
            gate_outputs = gate_outputs * mask[:, :, 0] + (1 - mask[:, :, 0]
                                                           ) * 1e3  #[B, T]
            gate_outputs = nn.Sigmoid()(gate_outputs)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs):
        embedded_inputs = paddle.tensor.transpose(self.embedding(inputs),
                                           [0, 2, 1])  #(B, C, T)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)  #[B, T, C]

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
