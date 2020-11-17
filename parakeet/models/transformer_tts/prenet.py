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

class PreNet(nn.Layer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout_rate=0.2,
                 bias=True):
        """Prenet before passing through the network.

        Args:
            input_size (int): the input channel size.
            hidden_size (int): the size of hidden layer in network.
            output_size (int): the output channel size.
            dropout_rate (float, optional): dropout probability. Defaults to 0.2.
        """
        super(PreNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        if bias:
            k = math.sqrt(1.0 / input_size)
            bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
                low=-k, high=k))
        else:
            bias_attr = False
        self.linear1 = nn.Linear(
            input_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            bias_attr=bias_attr)
        if bias:
            k = math.sqrt(1.0 / hidden_size)
            bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
                low=-k, high=k))
        else:
            bias_attr = False

        self.linear2 = nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            bias_attr=bias_attr)

    def forward(self, x):
        """
        Prepare network input.
        
        Args:
            x (Variable): shape(B, T, C), dtype float32, the input value.
                
        Returns:
            output (Variable): shape(B, T, C), the result after pernet.
        """
        x = F.dropout(
            F.relu(self.linear1(x)),
            self.dropout_rate)
        output = F.dropout(
            F.relu(self.linear2(x)),
            self.dropout_rate)
        return output
