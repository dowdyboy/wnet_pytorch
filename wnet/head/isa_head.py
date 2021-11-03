import torch
import torch.nn as nn
import torch.nn.functional as F

from wnet.backbone.component.basic_conv import BasicConv

import math


class ISAHead(nn.Module):

    def __init__(self, in_channel, out_channel, dropout_ratio=None):
        super(ISAHead, self).__init__()
        # 入口卷积
        self.conv101 = BasicConv(in_channel, in_channel, 1, 0, 1)
        # 定义ISA计算块
        self.isa = ISABlock(in_channel)
        # 出口卷积
        self.conv311 = BasicConv(2 * in_channel, in_channel, 3, 1, 1)
        # DROPOUT层
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio is not None else None
        # 分类用的卷积
        self.out_conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv101(x)
        isa_out = self.isa(x)
        x = torch.cat([x, isa_out], dim=1)
        x = self.conv311(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.out_conv(x)
        x = F.softmax(x, dim=1)
        return x


class ISABlock(nn.Module):

    def __init__(self, channel, down_factor=(8, 8)):
        super(ISABlock, self).__init__()
        # 分块大小
        self.down_factor = down_factor
        # 创建自注意力模块
        self.global_isa = SelfAttentionBlock(channel, channel, channel, channel)
        self.local_isa = SelfAttentionBlock(channel, channel, channel, channel)

    def forward(self, x):
        batch_size, c, h, w = x.size()
        loc_h, loc_w = self.down_factor
        # 根据分块大小，计算有多少块
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        # 因为上面取得是ceil，所以可能会有补足
        pad_h, pad_w = loc_h * glb_h - h, loc_w * glb_w - w
        if pad_h > 0 or pad_w > 0:
            # 对输入进行补足
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            x = F.pad(x, padding)
        # 将h、w进行拆解，拆解为块大小和块数量的乘积
        x = x.view(batch_size, c, glb_h, loc_h, glb_w, loc_w)
        # 调整位置，将远端的宽高放在最后两个维度
        x = x.permute(0, 3, 5, 1, 2, 4)  # batch_size, loc_h, loc_w, c, glb_h, glb_w
        x = x.reshape(-1, c, glb_h, glb_w)
        # 进行远端自注意力计算
        x = self.global_isa(x, x)

        # 同样的，进行近端自注意力计算
        x = x.view(batch_size, loc_h, loc_w, c, glb_h, glb_w)
        x = x.permute(0, 4, 5, 3, 1, 2)  # batch_size, glb_h, glb_w, c, loc_h, loc_w
        x = x.reshape(-1, c, loc_h, loc_w)
        x = self.local_isa(x, x)

        # 将位置调整为起始状态
        x = x.view(batch_size, glb_h, glb_w, c, loc_h, loc_w)  # batch_size, c, glb_h, loc_h, glb_w, loc_w
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(batch_size, c, glb_h * loc_h, glb_w * loc_w)
        # 去掉补足
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w]
        return x


class SelfAttentionBlock(nn.Module):

    def __init__(self,
                 key_in_channel,
                 query_in_channel,
                 key_query_out_channel,
                 out_channel,
                 key_query_num_conv=2,
                 value_num_conv=2,
                 key_query_norm=True,
                 value_query_norm=False,
                 matmul_norm=True):
        super(SelfAttentionBlock, self).__init__()
        self.key_in_channel = key_in_channel
        self.query_in_channel = query_in_channel
        self.key_query_out_channel = key_query_out_channel
        self.out_channel = out_channel
        self.matmul_norm = matmul_norm
        self.key_project = self.build_project(key_in_channel, key_query_out_channel, key_query_num_conv, key_query_norm)
        self.query_project = self.build_project(query_in_channel, key_query_out_channel, key_query_num_conv, key_query_norm)
        self.value_project = self.build_project(key_in_channel, out_channel, value_num_conv, value_query_norm)
        self.out_project = self.build_project(out_channel, out_channel, 1, True)

    def build_project(self, in_channel, out_channel, num_conv, use_conv_module):
        if use_conv_module:
            convs = [
                BasicConv(in_channel, out_channel, 1, 0, 1)
            ]
            for _ in range(num_conv - 1):
                convs.append(BasicConv(out_channel, out_channel, 1, 0, 1))
        else:
            convs = [nn.Conv2d(in_channel, out_channel, 1)]
            for _ in range(num_conv - 1):
                convs.append(nn.Conv2d(out_channel, out_channel, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_x, key_x):
        batch_size = query_x.size(0)
        query = self.query_project(query_x)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_x)
        value = self.value_project(key_x)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.key_query_out_channel ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_x.shape[2:])
        out = self.out_project(context)
        return out
