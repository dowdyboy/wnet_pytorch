import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class XGZLDataset(Dataset):

    def __init__(self, im_dir, label_dir, im_size=(512, 512)):
        super(XGZLDataset, self).__init__()
        self.im_size = im_size
        self.im_dir = im_dir
        self.label_dir = label_dir
        self.im_filename_list = os.listdir(self.im_dir)
        self.label_filename_list = os.listdir(self.label_dir)

    def __getitem__(self, idx):
        im_raw = Image.open(os.path.join(self.im_dir, self.im_filename_list[idx]))
        label_raw = Image.open(os.path.join(self.label_dir, self.im_filename_list[idx])).convert('L')
        # print(os.path.join(self.im_dir, self.im_filename_list[idx]), os.path.join(self.label_dir, self.im_filename_list[idx]))
        im = np.array(im_raw)
        # label = np.array(label_raw)
        im_h, im_w, _ = im.shape
        new_h, new_w, gap_h, gap_w = 0, 0, 0, 0
        if im_w > im_h:
            new_w = self.im_size[1]
            new_h = int(new_w / float(im_w) * im_h)
            gap_h = (self.im_size[0] - new_h) // 2
        else:
            new_h = self.im_size[0]
            new_w = int(new_h / float(im_h) * im_w)
            gap_w = (self.im_size[1] - new_w) // 2

        im_raw = im_raw.resize((new_w, new_h), Image.BICUBIC)
        new_im_raw = Image.new('RGB', (self.im_size[1], self.im_size[0]), (128, 128, 128))
        new_im_raw.paste(im_raw, (gap_w, gap_h))

        label_raw = label_raw.resize((new_w, new_h), Image.BICUBIC)
        new_label_raw = Image.new('L', (self.im_size[1], self.im_size[0]), 0)
        new_label_raw.paste(label_raw, (gap_w, gap_h))
        new_label = np.array(new_label_raw)
        new_label[new_label > 128] = 1
        new_label[new_label <= 128] = 0
        return torch.from_numpy(np.array(new_im_raw)).permute(2, 1, 0).contiguous(), torch.from_numpy(new_label).permute(1, 0).contiguous()

    def __len__(self):
        return len(self.im_filename_list)
