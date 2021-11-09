import torch
import torch.nn as nn

from wnet.config import WNetConfig
from wnet.backbone.component.basic_conv import BasicConv
from wnet.backbone.component.d_fuse_block import DFuseBlock
from wnet.backbone.component.stem_block import StemBlock
from wnet.backbone.component.basic_res_block import BasicResBlock
from wnet.backbone.component.spp import SPP
from wnet.backbone.component.s_fuse_block import SFuseBlock
from wnet.backbone.component.lr_res_block import LRResBlock
from wnet.backbone.component.cross_agg_layer import CrossAggLayer
from wnet.head.isa_head import ISAHead


class ULikeNet(nn.Module):

    def __init__(self, conf: WNetConfig):
        super(ULikeNet, self).__init__()
        self.d_conv1_312 = BasicConv(3, 64, 3, 1, 2)
        self.d_conv2_311 = BasicConv(64, 64, 3, 1, 1)
        self.d_conv3_312 = BasicConv(64, 128, 3, 1, 2)
        self.d_conv4_311 = nn.Sequential(
            BasicConv(128, 128, 3, 1, 1),
            BasicConv(128, 128, 3, 1, 1)
        )
        self.d_conv5_312 = BasicConv(128, 256, 3, 1, 2)
        self.d_conv6_311 = nn.Sequential(
            BasicConv(256, 256, 3, 1, 1),
            BasicConv(256, 256, 3, 1, 1)
        )
        self.d_tconv7_212 = nn.ConvTranspose2d(256, 128, kernel_size=2, padding=0, stride=2)
        self.d_fuse8 = DFuseBlock(128)
        self.d_tconv9_212 = nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2)
        self.d_fuse10 = DFuseBlock(64)
        self.d_conv11_311 = BasicConv(64, 64, 3, 1, 1)

        self.s_stem1 = StemBlock(3, 16)
        self.s_res2 = BasicResBlock(16)
        self.s_conv3_312 = BasicConv(16, 32, 3, 1, 2)
        self.s_res4 = BasicResBlock(32)
        self.s_conv5_312 = BasicConv(32, 64, 3, 1, 2)
        self.s_res6 = nn.Sequential(
            BasicResBlock(64),
            BasicResBlock(64)
        )
        self.s_conv7_312 = BasicConv(64, 128, 3, 1, 2)
        self.s_res8 = nn.Sequential(
            BasicResBlock(128),
            BasicResBlock(128)
        )
        self.s_spp9 = SPP(128)
        self.s_tconv10_212 = nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2)
        self.s_fuse11 = SFuseBlock(64)
        self.s_tconv12_212 = nn.ConvTranspose2d(64, 32, kernel_size=2, padding=0, stride=2)
        self.s_fuse13 = SFuseBlock(32)
        self.s_conv14_311 = BasicConv(32, 64, 3, 1, 1)
        self.s_lrres1 = LRResBlock(32)
        self.s_lrres2 = LRResBlock(64)

        self.cross_agg = CrossAggLayer(64)
        self.head_upsample = nn.Upsample(scale_factor=2)
        self.head = ISAHead(64, conf.num_class)

    def forward(self, x):
        # 走细节分支
        d_out = self.d_conv1_312(x)
        d_fuse_out1 = self.d_conv2_311(d_out)
        d_out = self.d_conv3_312(d_fuse_out1)
        d_fuse_out2 = self.d_conv4_311(d_out)
        d_out = self.d_conv5_312(d_fuse_out2)
        d_out = self.d_conv6_311(d_out)
        d_out = self.d_tconv7_212(d_out)
        d_out = self.d_fuse8(d_out, d_fuse_out2)
        d_out = self.d_tconv9_212(d_out)
        d_out = self.d_fuse10(d_out, d_fuse_out1)
        d_out = self.d_conv11_311(d_out)

        # 走语义分支
        s_out = self.s_stem1(x)
        s_out = self.s_res2(s_out)
        s_out = self.s_conv3_312(s_out)
        s_out = self.s_res4(s_out)
        s_fuse_out1 = self.s_lrres1(s_out)
        s_out = self.s_conv5_312(s_out)
        s_out = self.s_res6(s_out)
        s_fuse_out2 = self.s_lrres2(s_out)
        s_out = self.s_conv7_312(s_out)
        s_out = self.s_res8(s_out)
        s_out = self.s_spp9(s_out)
        s_out = self.s_tconv10_212(s_out)
        s_out = self.s_fuse11(s_out, s_fuse_out2)
        s_out = self.s_tconv12_212(s_out)
        s_out = self.s_fuse13(s_out, s_fuse_out1)
        s_out = self.s_conv14_311(s_out)

        out = self.cross_agg(d_out, s_out)
        out = self.head_upsample(out)
        out = self.head(out)
        return out
