import torch
import torch.nn as nn
from wnet.backbone.component.basic_conv import BasicConv


class CrossAggLayer(nn.Module):

    def __init__(self, channel):
        super(CrossAggLayer, self).__init__()
        # 深度可分离卷积(3,1,1)
        self.depth_conv_1 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.point_conv_1 = nn.Conv2d(channel, channel, 1, 1, 0, groups=1)

        self.depth_conv_2 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.point_conv_2 = nn.Conv2d(channel, channel, 1, 1, 0, groups=1)

        self.depth_conv_3 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.point_conv_3 = nn.Conv2d(channel, channel, 1, 1, 0, groups=1)

        self.bn_1 = nn.BatchNorm2d(channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.bn_3 = nn.BatchNorm2d(channel)
        self.bn_4 = nn.BatchNorm2d(channel)
        self.bn_5 = nn.BatchNorm2d(channel)
        self.conv101 = BasicConv(channel, channel, 1, 0, 1)
        self.conv2d311 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.up_sample_1 = nn.Upsample(scale_factor=4)
        self.up_sample_2 = nn.Upsample(scale_factor=4)
        self.conv2d312 = nn.Conv2d(channel, channel, 3, 2, 1)
        self.avg_pooling = nn.AvgPool2d(3, 2, 1)
        self.conv2d101 = nn.Conv2d(channel, channel, 1, 1, 0)
        self.conv311 = BasicConv(channel, channel, 3, 1, 1)

    def forward(self, x_detail, x_semi):
        # 左1分支
        l11 = self.depth_conv_1(x_detail)
        l12 = self.point_conv_1(l11)
        l13 = self.bn_1(l12)
        l14 = self.depth_conv_2(l13)
        l15 = self.point_conv_2(l14)
        l16 = self.bn_2(l15)
        l17 = self.conv101(l16)

        # 左2分支
        l21 = self.conv2d311(x_semi)
        l22 = self.bn_3(l21)
        l23 = self.up_sample_1(l22)
        m = nn.Sigmoid()
        l24 = m(l23)

        # 左分支合并
        l = torch.mul(l17, l24)

        # 右1分支
        r11 = self.conv2d312(x_detail)
        r12 = self.bn_4(r11)
        r13 = self.avg_pooling(r12)

        # 右2分支
        r21 = self.depth_conv_3(x_semi)
        r22 = self.point_conv_3(r21)
        r23 = self.bn_5(r22)
        r24 = self.conv2d101(r23)
        f = nn.Sigmoid()
        r25 = f(r24)

        # 右分支合并
        r = torch.mul(r13, r25)
        r = self.up_sample_2(r)

        # 左右合并
        z = l + r
        out = self.conv311(z)
        return out
