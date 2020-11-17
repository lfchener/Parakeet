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
from paddle import nn
import paddle.nn.functional as F
import time

class ConvNorm(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True,
                 data_format='NCL'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        if bias:
            k = math.sqrt(1.0 / in_channels)
            bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
                low=-k, high=k))
        self.conv = nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias_attr=bias_attr,
            data_format=data_format)

    def forward(self, input):
        conv = self.conv(input)
        return conv


class Encoder(nn.Layer):
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
                dilation=1,
                data_format='NLC')
            batchnorm = nn.BatchNorm1D(embedding_dim, data_format='NLC')
            self.convolutions.append(conv)
            self.batchnorms.append(batchnorm)
            self.add_sublayer("convolutions_{}".format(i), conv)
            self.add_sublayer("batchnorms{}".format(i), batchnorm)

        self.hidden_size = int(embedding_dim / 2)
        self.lstm = nn.LSTM(
            embedding_dim, self.hidden_size, direction="bidirectional")

    def forward(self, x, input_lens):
        # x.shape = [B, T, C]

        for conv, batchnorm in zip(self.convolutions, self.batchnorms):
            x = F.dropout(F.relu(batchnorm(conv(x))), 0.0)  #(B, T, C)  0.5

        #x = paddle.transpose(x, [0, 2, 1])  # (B, T, C)
        batch_size = x.shape[0]

        pre_hidden = paddle.zeros(
            shape=[2, batch_size, self.hidden_size], dtype=x.dtype)
        pre_cell = paddle.zeros(
            shape=[2, batch_size, self.hidden_size], dtype=x.dtype)

        output, _ = self.lstm(
            inputs=x,
            initial_states=(pre_hidden, pre_cell),
            sequence_length=input_lens)

        return output

    def inference(self, x):
        for conv, batchnorm in zip(self.convolutions, self.batchnorms):
            x = F.dropout(F.relu(batchnorm(conv(x))), 0.5)  #(B, C, T)

        x = paddle.transpose(x, [0, 2, 1])  # (B, T, C)

        lstm_forward = self.lstm_forward(x)
        lstm_reverse = self.lstm_reverse(x)

        output = paddle.concat([lstm_forward, lstm_reverse], axis=-1)

        return output
