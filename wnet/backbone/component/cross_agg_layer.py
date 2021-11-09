import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv


class CrossAggLayer(nn.Module):

    def __init__(self, channel):
        super(CrossAggLayer, self).__init__()
        # 深度可分离卷积(3,1,1)
        self.depth_conv = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.point_conv = nn.Conv2d(channel, channel, 1, 1, 0, groups=1)

        self.bn = nn.BatchNorm2d(channel)
        self.conv101 = BasicConv(1, 0, 1)
        self.conv2d311 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.up_sample = nn.Upsample(scale_factor=4)
        self.conv2d312 = nn.Conv2d(channel, channel, 3, 2, 1)
        self.avg_pooling = nn.AvgPool2d(3, 2, 1)
        self.conv2d101 = nn.Conv2d(channel, channel, 1, 1, 0)
        self.conv311 = BasicConv(channel, channel, 3, 1, 1)

    def forward(self, x_detail, x_semi):
        # 左1分支
        l11 = self.depth_conv(x_detail)
        l12 = self.point_conv(l11)
        l13 = self.bn(l12)
        l14 = self.depth_conv(l13)
        l15 = self.point_conv(l14)
        l16 = self.bn(l15)
        l17 = self.conv101(l16)

        # 左2分支
        l21 = self.conv2d311(x_semi)
        l22 = self.bn(l21)
        l23 = self.up_sample(l22)
        l24 = nn.Sigmoid(l23)

        # 左分支合并
        l = torch.mul(l17, l24)

        # 右1分支
        r11 = self.conv2d312(x_detail)
        r12 = self.bn(r11)
        r13 = self.avg_pooling(r12)

        # 右2分支
        r21 = self.depth_conv(x_semi)
        r22 = self.point_conv(r21)
        r23 = self.bn(r22)
        r24 = self.conv2d101(r23)
        r25 = nn.Sigmoid(r24)

        # 右分支合并
        r = torch.mul(r13, r25)
        r = self.up_sample(r)

        # 左右合并
        z = l + r
        out = self.conv311(z)
        return out



