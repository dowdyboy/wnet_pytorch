import torch
import torch.nn as nn

from wnet.config import WNetConfig


class ULikeNet(nn.Module):

    def __init__(self, conf: WNetConfig):
        super(ULikeNet, self).__init__()

    def forward(self, x):
        pass
