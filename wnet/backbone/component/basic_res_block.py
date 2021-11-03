import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv


class BasicResBlock(nn.Module):
    
    def __init__(self, channel):
        super(BasicResBlock, self).__init__()
        self.layer1 = BasicConv(channel, channel, 3, 1, 1)
        self.layer2 = BasicConv(channel, channel, 3, 1, 1)
        self.layer3 = BasicConv(channel, channel, 1, 0, 1)

    def forward(self, x1):
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = x1 + x4
        return x5

