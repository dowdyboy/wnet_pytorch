import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv


class StemBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(StemBlock, self).__init__()
        self.conv312_1 = BasicConv(in_channel, out_channel, 3, 1, 2)
        self.conv312_2 = BasicConv(out_channel//2, out_channel, 3, 1, 2)
        self.conv101 = BasicConv(out_channel, out_channel//2, 1, 0, 1)
        self.max_pooling = nn.MaxPool2d(3, 2, 1)
        self.conv311 = BasicConv(2*out_channel, out_channel, 3, 1, 1)

    def forward(self, x1):
        x2 = self.conv312_1(x1)
        y1 = self.conv101(x2)
        y2 = self.conv312_2(y1)
        z1 = self.max_pooling(x2)
        mix = torch.cat([y2, z1], dim=1)

        output = self.conv311(mix)
        return output



