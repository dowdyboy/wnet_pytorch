import torch
import torch.nn as nn

import numpy as np


class WNetLoss(nn.Module):
    
    def __init__(self):
        super(WNetLoss, self).__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, x, gt):
        batch_size, _, h, w = x.size()
        cel_loss = self.cel(x, gt)
        area = x.size(-1) * x.size(-2) * x.size(0)
        trans_x = ((x[:, 1, :, :] > x[:, 0, :, :]) * 1)
        area_loss = torch.square(torch.sum(trans_x) / area - torch.sum(gt) / area)

        x_pos_all = []
        for i in range(batch_size):
            x_pos = torch.where(trans_x[i] == 1)
            x_x_min, x_x_max, x_y_min, x_y_max = torch.min(x_pos[0]), torch.max(x_pos[0]), torch.min(x_pos[1]), torch.max(x_pos[1])
            x_pos_all.append([x_x_min, x_y_min, x_x_max, x_y_max])
        x_pos_all = torch.tensor(x_pos_all, dtype=torch.float)
        x_pos_center_x = (x_pos_all[:, 0] + x_pos_all[:, 2]) / 2 / w
        x_pos_center_y = (x_pos_all[:, 1] + x_pos_all[:, 3]) / 2 / h
        x_pos_length = torch.square((x_pos_all[:, 2] - x_pos_all[:, 0]) / w) + torch.square((x_pos_all[:, 3] - x_pos_all[:, 1]) / h)
        # print(x_pos_center_x, x_pos_center_y)
        # print(x_pos_length)

        gt_pos_all = []
        for i in range(batch_size):
            gt_pos = torch.where(gt[i] == 1)
            gt_x_min, gt_x_max, gt_y_min, gt_y_max = torch.min(gt_pos[0]), torch.max(gt_pos[0]), torch.min(gt_pos[1]), torch.max(gt_pos[1])
            gt_pos_all.append([gt_x_min, gt_y_min, gt_x_max, gt_y_max])
        gt_pos_all = torch.tensor(gt_pos_all, dtype=torch.float)
        gt_pos_center_x = (gt_pos_all[:, 0] + gt_pos_all[:, 2]) / 2 / w
        gt_pos_center_y = (gt_pos_all[:, 1] + gt_pos_all[:, 3]) / 2 / h
        gt_pos_length = torch.square((gt_pos_all[:, 2] - gt_pos_all[:, 0]) / w) + torch.square((gt_pos_all[:, 3] - gt_pos_all[:, 1]) / h)
        # print(gt_pos_center_x, gt_pos_center_y)
        # print(gt_pos_length)

        dist_c = torch.square(x_pos_center_x - gt_pos_center_x) + torch.square(x_pos_center_y - gt_pos_center_y)
        dist_c = torch.sum(dist_c) / batch_size
        dist_length = torch.square(x_pos_length - gt_pos_length)
        dist_length = torch.sum(dist_length) / batch_size

        return cel_loss + area_loss + 0.5 * dist_c + 0.5 * dist_length
