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

import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from .customized import Conv1D


class LocationLayer(dg.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = Conv1D(
            2,
            attention_n_filters,
            attention_kernel_size,
            1,
            padding,
            1,
            bias_attr=False)
        self.location_dense = dg.Linear(
            attention_n_filters, attention_dim, bias_attr=False)

    def forward(self, attention_weights_cat):
        # attention_weights_cat.shape=[B, 2, T]
        processed_attention = self.location_conv(
            attention_weights_cat)  #[B, C, T]
        processed_attention = layers.transpose(processed_attention,
                                               [0, 2, 1])  #[B, T, C]
        processed_attention = self.location_dense(
            processed_attention)  #[B, T, C]
        return processed_attention


class LocationSensitiveAttention(dg.Layer):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = dg.Linear(
            attention_rnn_dim, attention_dim, bias_attr=False)
        self.memory_layer = dg.Linear(
            embedding_dim, attention_dim, bias_attr=False)
        self.value = dg.Linear(attention_dim, 1, bias_attr=False)
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
            layers.unsqueeze(
                attention_hidden_state, axes=[1]))  #[B, 1, C]
        processed_attention_weights = self.location_layer(
            attention_weights_cat)  #[B, T, C]
        processed_memory = self.memory_layer(memory)
        energies = self.value(
            layers.tanh(processed_query + processed_attention_weights +
                        processed_memory))  #[B, T, 1]

        alignment = layers.squeeze(energies, axes=[-1])  # [B, T]

        if mask is not None:
            alignment = alignment + mask * -1e30  # [B, T]

        attention_weights = layers.softmax(alignment)  #[B, T]
        attention_context = layers.matmul(
            layers.unsqueeze(
                attention_weights, axes=[1]), memory)  #[B, 1, C]
        attention_context = layers.squeeze(
            attention_context, axes=[1])  # [B, C]

        return attention_context, attention_weights
