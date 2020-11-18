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


class LocationLayer(nn.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = nn.Conv1D(
            2,
            attention_n_filters,
            attention_kernel_size,
            1,
            padding,
            1,
            bias_attr=False)
        self.location_dense = nn.Linear(
            attention_n_filters, attention_dim, bias_attr=False)

    def forward(self, attention_weights_cat):
        # attention_weights_cat.shape=[B, 2, T]
        processed_attention = self.location_conv(
            attention_weights_cat)  #[B, C, T]
        processed_attention = paddle.transpose(processed_attention,
                                               [0, 2, 1])  #[B, T, C]
        processed_attention = self.location_dense(
            processed_attention)  #[B, T, C]
        return processed_attention


class LocationSensitiveAttention(nn.Layer):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = nn.Linear(
            attention_rnn_dim, attention_dim, bias_attr=False)
        self.memory_layer = nn.Linear(
            embedding_dim, attention_dim, bias_attr=False)
        self.value = nn.Linear(attention_dim, 1, bias_attr=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)

    def forward(self,
                attention_hidden_state,
                memory,
                attention_weights_cat,
                mask=None):
        # attention_hidden_state.shape = [B, C]
        # memory.shape = [B, T, C]
        # attention_weights_cat.shape = [B, 2, T]
        # mask.shape = [B, T]

        # get alignment energies
        processed_query = self.query_layer(
                paddle.unsqueeze(attention_hidden_state, axis=[1]))  #[B, 1, C]
        processed_attention_weights = self.location_layer(
            attention_weights_cat)  #[B, T, C]
        processed_memory = self.memory_layer(memory) #(B, T, C)
        energies = self.value(
            paddle.tanh(processed_query + processed_attention_weights +
                        processed_memory))  #[B, T, 1]

        if mask is not None:
            alignment = energies + mask * -1e9  # [B, T, 1]

        attention_weights = F.softmax(alignment, axis=1)  #[B, T, 1]
        attention_context = paddle.matmul(
                attention_weights, memory, transpose_x=True)  #[B, 1, C]
        attention_weights = paddle.squeeze(attention_weights, axis=[-1]) #[B, C]
        attention_context = paddle.squeeze(attention_context, axis=[1]) #[B, C]

        return attention_context, attention_weights
