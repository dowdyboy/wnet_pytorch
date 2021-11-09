import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv


class SFuseBlock(nn.Module):

    def __init__(self, channel):
        super(SFuseBlock, self).__init__()
        self.depth_conv_1 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.point_conv_1 = nn.Conv2d(channel, channel, 1, 1, 0, groups=1)
        self.depth_conv_2 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.point_conv_2 = nn.Conv2d(channel, channel, 1, 1, 0, groups=1)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.conv101 = BasicConv(2*channel, channel, 1, 0, 1)

    def forward(self, x1, y1):
        x2 = self.depth_conv_1(x1)
        x3 = self.point_conv_1(x2)
        x4 = self.bn_1(x3)
        y2 = self.depth_conv_2(y1)
        y3 = self.point_conv_2(y2)
        y4 = self.bn_2(y3)
        z = torch.cat((x4, y4), dim=1)
        output = self.conv101(z)
        return output

