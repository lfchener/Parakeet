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
import paddle.fluid.layers as layers


class DynamicLSTM(dg.Layer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation=layers.sigmoid,
                 activation=layers.tanh,
                 forget_bias=1.0,
                 use_cudnn_impl=True,
                 dtype='float32'):
        super(DynamicLSTM, self).__init__()
        self.lstm_unit = dg.LSTMCell(
            hidden_size,
            input_size,
            param_attr=param_attr,
            bias_attr=bias_attr,
            gate_activation=gate_activation,
            activation=activation,
            forget_bias=forget_bias,
            use_cudnn_impl=use_cudnn_impl,
            dtype=dtype)
        self.hidden_size = hidden_size
        self.is_reverse = is_reverse

    def forward(self, inputs, input_lens=None):
        """
        Dynamic LSTM block.
        
        Args:
            input (Variable): shape(B, T, C), dtype float32, the input value.
                 
        Returns:
            output (Variable): shape(B, T, C), the result compute by LSTM.
        """
        batch_size = inputs.shape[0]
        pre_hidden = layers.zeros(
            shape=[batch_size, self.hidden_size], dtype=inputs.dtype)
        pre_cell = layers.zeros(
            shape=[batch_size, self.hidden_size], dtype=inputs.dtype)
        if input_lens is not None:
            masks = layers.unsqueeze(
                layers.sequence_mask(input_lens), axes=[-1])  #[B, T, 1]
        res = []
        for i in range(inputs.shape[1]):
            # Note: padding reverse!!!
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = inputs[:, i, :]
            hidden, cell = self.lstm_unit(input_, pre_hidden, pre_cell)
            if input_lens is not None:
                mask = masks[:, i, :]
                hidden = hidden * mask + pre_hidden * (1 - mask)
                cell = cell * mask + pre_cell * (1 - mask)
            hidden_ = layers.reshape(hidden, [-1, 1, hidden.shape[1]])
            res.append(hidden_)
            pre_hidden = hidden
            pre_cell = cell
        if self.is_reverse:
            res = res[::-1]

        res = layers.concat(res, axis=1)
        return res
