import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv


class DFuseBlock(nn.Module):

    def __init__(self, channel):
        super(DFuseBlock, self).__init__()
        self.conv311_1to1 = BasicConv(channel, channel, 3, 1, 1)
        self.conv311_2to1 = BasicConv(2 * channel, 2 * channel, 3, 1, 1)
        self.conv101 = BasicConv(2*channel, channel, 1, 0, 1)

    def forward(self, x1, y1):
        x2 = self.conv311_1to1(x1)
        y2 = self.conv311_1to1(y1)
        z1 = torch.cat((x2, y2), dim=1)
        z2 = self.conv311_2to1(z1)
        z3 = self.conv101(z2)
        return z3

