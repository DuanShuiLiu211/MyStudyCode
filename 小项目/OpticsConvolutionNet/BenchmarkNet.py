import torch
from typing import Optional
from torch import nn, Tensor
from collections import OrderedDict


class BenchmarkNet(nn.Module):
    def __init__(self, mode : Optional = "tiny_cnn"):
        super().__init__()
        self.mode = mode
        if mode == "tiny_cnn":
            self.tiny_cnn = torch.nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(1, 9, 5)),
                                                             ('conv1_2', nn.Conv2d(9, 1, 1)),
                                                             ('conv2_1', nn.Conv2d(1, 25, 5)),
                                                             ('conv2_2', nn.Conv2d(25, 1, 1)),
                                                             ('conv3_1', nn.Conv2d(1, 49, 5)),
                                                             ('conv3_1', nn.Conv2d(49, 1, 5)),
                                                             ('conv4_1', nn.Conv2d(1, 81, 5)),
                                                             ('conv4_2', nn.Conv2d(81, 1, 5)),
                                                             ('conv5_1', nn.Conv2d(1, 49, 5)),
                                                             ('conv5_1', nn.Conv2d(49, 1, 5)),
                                                             ('conv6_1', nn.Conv2d(1, 25, 5)),
                                                             ('conv6_2', nn.Conv2d(25, 1, 1)),
                                                             ('conv7_1', nn.Conv2d(1, 9, 5)),
                                                             ('conv7_2', nn.Conv2d(9, 1, 1)),]
                                                            )
                                                )
        else:
            print("no model to initialize")

    def forward(self, inputs : Tensor) -> Tensor:
        if self.mode == "tiny_cnn":
            outputs = self.tiny_cnn(inputs)
        else:
            outputs = inputs
            print("nothing going on")

        return outputs
