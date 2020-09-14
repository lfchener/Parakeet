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

import math
import paddle
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from parakeet.modules.customized import Conv1D
from parakeet.modules.dynamic_lstm import DynamicLSTM


class ConvNorm(dg.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        if bias:
            k = math.sqrt(1.0 / in_channels)
            bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k))
        self.conv = Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias_attr=bias_attr)

    def forward(self, input):
        conv = self.conv(input)
        return conv


class Encoder(dg.Layer):
    def __init__(self, n_convs, embedding_dim, kernel_size):
        super(Encoder, self).__init__()
        self.convolutions = []
        self.batchnorms = []
        for i in range(n_convs):
            conv = ConvNorm(
                embedding_dim,
                embedding_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1)
            batchnorm = dg.BatchNorm(embedding_dim)
            self.convolutions.append(conv)
            self.batchnorms.append(batchnorm)
            self.add_sublayer("convolutions_{}".format(i), conv)
            self.add_sublayer("batchnorms{}".format(i), batchnorm)
        self.dropout = dg.Dropout(
            p=0.5, dropout_implementation='upscale_in_train')

        self.hidden_size = int(embedding_dim / 2)
        self.lstm = paddle.nn.LSTM(
            embedding_dim, self.hidden_size, direction="bidirectional")

    def forward(self, x, input_lens):
        # x.shape = [B, C, T]

        for conv, batchnorm in zip(self.convolutions, self.batchnorms):
            x = self.dropout(layers.relu(batchnorm(conv(x))))  #(B, C, T)

        x = layers.transpose(x, [0, 2, 1])  # (B, T, C)
        batch_size = x.shape[0]

        pre_hidden = layers.zeros(
            shape=[2, batch_size, self.hidden_size], dtype=x.dtype)
        pre_cell = layers.zeros(
            shape=[2, batch_size, self.hidden_size], dtype=x.dtype)

        output, _ = self.lstm(
            inputs=x,
            initial_states=(pre_hidden, pre_cell),
            sequence_length=input_lens)

        mask = layers.unsqueeze(
            layers.sequence_mask(
                input_lens, maxlen=output.shape[1]),
            axes=[-1])  #(B, T, 1)
        output = output * mask

        return output

    def inference(self, x):
        for conv, batchnorm in zip(self.convolutions, self.batchnorms):
            x = self.dropout(layers.relu(batchnorm(conv(x))))  #(B, C, T)

        x = layers.transpose(x, [0, 2, 1])  # (B, T, C)

        lstm_forward = self.lstm_forward(x)
        lstm_reverse = self.lstm_reverse(x)

        output = layers.concat(input=[lstm_forward, lstm_reverse], axis=-1)

        return output
