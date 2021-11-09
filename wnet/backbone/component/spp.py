import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv


class SPP(nn.Module):

    def __init__(self, channel):
        super(SPP, self).__init__()
        self.conv311_1to1_1 = BasicConv(channel, channel, 3, 1, 1)
        self.conv311_1to1_2 = BasicConv(channel, channel, 3, 1, 1)
        self.mp311 = nn.MaxPool2d(3, 1, 1)
        self.mp521 = nn.MaxPool2d(5, 1, 2)
        self.mp841 = nn.MaxPool2d(9, 1, 4)
        self.conv311_4to1 = BasicConv(4*channel, channel, 3, 1, 1)

    def forward(self, x1):
        x2 = self.conv311_1to1_1(x1)
        y1 = self.mp311(x2)
        y2 = self.mp521(x2)
        y3 = self.mp841(x2)
        z1 = torch.cat((y1, y2, y3, x2), dim=1)
        z2 = self.conv311_4to1(z1)
        z3 = self.conv311_1to1_2(z2)
        return z3
