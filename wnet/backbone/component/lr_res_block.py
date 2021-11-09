import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv
from wnet.backbone.component.basic_res_block import BasicResBlock


class LRResBlock(nn.Module):

    def __init__(self, channel):
        super(LRResBlock, self).__init__()
        self.conv101_2to1 = BasicConv(channel, int(channel/2), 1, 0, 1)
        self.max_pooling = nn.MaxPool2d(3, 1, 1)
        self.basic_res = BasicResBlock(int(channel/2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(int(channel/2))
        self.conv101_1to1 = BasicConv(int(channel/2), int(channel/2), 1, 0, 1)
        self.conv311 = BasicConv(int(channel * 3 / 2), channel, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv101_2to1(x)
        w = self.max_pooling(x1)
        y = self.basic_res(x1)
        z = self.gap(x1)
        z = self.bn(z)
        z = self.conv101_1to1(z)
        z = z.repeat(1, 1, y.size()[2], y.size()[3])
        out = torch.cat((w, y, z), dim=1)
        out = self.conv311(out)
        return out


