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

import paddle
from paddle import nn
import paddle.nn.functional as F
from parakeet.models.transformer_tts.prenet import PreNet
from parakeet.modules.location_sensitive_attention import LocationSensitiveAttention
import time

class Decoder(nn.Layer):
    def __init__(self, n_mels, n_frames_per_step, encoder_embedding_dim,
                 prenet_dim, attention_rnn_dim, decoder_rnn_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 p_attention_dropout, p_decoder_dropout):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = PreNet(
            n_mels * n_frames_per_step,
            prenet_dim,
            prenet_dim,
            dropout_rate=0.0, #0.5
            bias=False)

        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim, attention_rnn_dim)

        self.attention_layer = LocationSensitiveAttention(
            attention_rnn_dim, encoder_embedding_dim, attention_dim,
            attention_location_n_filters, attention_location_kernel_size)
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim)
        self.linear_projection = nn.Linear(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mels * n_frames_per_step)
        self.gate_layer = nn.Linear(decoder_rnn_dim + encoder_embedding_dim, 1)

    def initialize_decoder_states(self, memory):
        batch_size = memory.shape[0]
        MAX_TIME = memory.shape[1]

        self.attention_hidden = paddle.zeros(
            shape=[batch_size, self.attention_rnn_dim], dtype=memory.dtype)
        self.attention_cell = paddle.zeros(
            shape=[batch_size, self.attention_rnn_dim], dtype=memory.dtype)

        self.decoder_hidden = paddle.zeros(
            shape=[batch_size, self.decoder_rnn_dim], dtype=memory.dtype)
        self.decoder_cell = paddle.zeros(
            shape=[batch_size, self.decoder_rnn_dim], dtype=memory.dtype)

        self.attention_weights = paddle.zeros(
            shape=[batch_size, MAX_TIME], dtype=memory.dtype)
        self.attention_weights_cum = paddle.zeros(
            shape=[batch_size, MAX_TIME], dtype=memory.dtype)
        self.attention_context = paddle.zeros(
            shape=[batch_size, self.encoder_embedding_dim], dtype=memory.dtype)

        self.memory = memory  #[B, T, C]

    def decode(self, decoder_input):
        # decoder_input.shape=[B, C]
        cell_input = paddle.concat(
            [decoder_input, self.attention_context], axis=-1)  #[B, C]
        
        _, (self.attention_hidden, self.attention_cell) = self.attention_rnn(
            cell_input, (self.attention_hidden,
                         self.attention_cell))  #[B, C], [B, C]
        
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout)  #[B, C]
        attention_weights_cat = paddle.stack([self.attention_weights, self.attention_weights_cum], axis=1)  #[B, 2, T]

        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, attention_weights_cat,
            self.mask)  #[B, C], [B, T]

        self.attention_weights_cum += self.attention_weights  #[B, T]
        decoder_input = paddle.concat(
            [self.attention_hidden, self.attention_context], axis=-1)  #[B, 2C]
        a, (self.decoder_hidden, self.decoder_cell) = self.decoder_rnn(
            decoder_input, (self.decoder_hidden,
                            self.decoder_cell))  #[B, C] [B, C]
        self.decoder_hidden = F.dropout(self.decoder_hidden, p = self.p_decoder_dropout)  #[B, C]

        decoder_hidden_attention_context = paddle.concat(
            [self.decoder_hidden, self.attention_context], axis=-1)  #[B, 2C]
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)  #[B, C]

        gate_prediction = self.gate_layer(
            decoder_hidden_attention_context)  #[B, 1]
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lens):
        # memory.shape=[B, T, C], decoder_inputs.shape=[B, T, C]
        batch_size = decoder_inputs.shape[0]
        n_feature = decoder_inputs.shape[-1]
        decoder_inputs = paddle.concat(
            [
                paddle.zeros(
                    shape=[batch_size, 1, n_feature],
                    dtype=decoder_inputs.dtype), decoder_inputs
            ],
            axis=1)
        decoder_inputs = self.prenet(decoder_inputs)  #(B, T, C)

        # mel 频谱每个 time step 的 mask，所以少了 T_mel 维
        self.mask = paddle.cast(memory[:,:,0:1] == 0, dtype=memory.dtype) #(B, T, 1)
        self.initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = [], [], []

        while len(mel_outputs) < decoder_inputs.shape[1] - 1:  # 除去最后一个时间步
            decoder_input = decoder_inputs[:, len(mel_outputs), :]  #[B, C]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)  #[B, C], [B, 1], [B, T]
            mel_outputs += [mel_output]  # T 个 [B, C]
            gate_outputs += [gate_output]  #T 个 [B, 1]
            alignments += [attention_weights]  # T 个 [B, T]

        alignments = paddle.stack(alignments, axis=1)  #[B, T, T]
        gate_outputs = paddle.concat(gate_outputs, axis=1)  #[B, T]
        mel_outputs = paddle.stack(mel_outputs, axis=1)  #[B, T, C]

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, gate_threshold=0.5, max_decoder_steps=1000):
        batch_size = memory.shape[0]
        decoder_input = paddle.zeros(
            shape=[batch_size, self.n_mel_channels * self.n_frames_per_step],
            dtype=memory.dtype)  #[B, C]

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output]
            gate_outputs += [paddle.squeeze(
                gate_output, axes=[1])]  #！！！！diff！！！
            alignments += [alignment]

            if F.sigmoid(gate_output) > gate_threshold:
                break
            elif len(mel_outputs) == max_decoder_steps:
                print("Warning! Reached max decoder steps!!!")
                break

            decoder_input = mel_output

        alignments = paddle.stack(alignments, axis=1)  #[B, T]
        gate_outputs = paddle.stack(gate_outputs, axis=1)  #[B, T]
        mel_output = paddle.stack(mel_output, axis=1)  #[B, T, C]
        mel_output = paddle.reshape(
            mel_output, shape=[0, -1, self.n_mel_channels])  #[B, T, C]

        return mel_outputs, gate_outputs, alignments
