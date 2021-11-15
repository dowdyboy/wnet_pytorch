import torch
import torch.nn
from torch.utils.data import DataLoader

from wnet.config import WNetConfig
from wnet.backbone.u_like_net import ULikeNet
from wnet.data.xgzl_dataset import XGZLDataset
from utils.log import MyNetLogger

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import os

if __name__ == '__main__':
    conf = WNetConfig()
    conf.set_cuda(True)
    conf.set_class_config(class_list=['background', 'zl'])
    conf.set_input_shape(256, 256)
    conf.set_num_worker(0)
    conf.set_log('logs/eval.log')
    conf.set_pretrained_path('checkpoints/epoch12')

    logger = print if conf.log_file is None else MyNetLogger.default(conf.log_file)
    logger(conf)

    device = 'cuda' if conf.use_cuda else 'cpu'

    model = ULikeNet(conf)
    model.to(device)

    test_dataset = XGZLDataset('datasets/xgzl/images/validation', 'datasets/xgzl/annotations/validation-old', conf.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=conf.num_worker)

    out_mask_dir = 'out'
    if not os.path.isdir(out_mask_dir):
        os.makedirs(out_mask_dir)

    if conf.pretrain_path is not None:
        filename_list = os.listdir(conf.pretrain_path)
        model_filename = None
        for filename in filename_list:
            if filename.find('model') > -1:
                model_filename = filename
        if model_filename is not None:
            model.load_state_dict(torch.load(os.path.join(conf.pretrain_path, model_filename)))
        logger('successfully load pretrained : {}'.format(conf.pretrain_path))

    model.eval()
    for bat_im, bat_label, filename in test_loader:
        bat_im, bat_label = bat_im.to(device), bat_label.to(device)
        out = model(bat_im)
        prob, cls = torch.max(out[0], dim=0)

        cls_view = cls.permute(1, 0).cpu().numpy()
        cls_view[cls_view == 1] = 255
        im = Image.fromarray(cls_view.astype(np.uint8), mode='L')
        im.save(os.path.join(out_mask_dir, filename[0]))
        # plt.figure()
        # plt.imshow(im, cmap='gray')
        # plt.show()
        # break
