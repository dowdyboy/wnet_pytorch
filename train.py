import torch
import torch.nn
from torch.utils.data import DataLoader

from utils.log import MyNetLogger
from wnet.config import WNetConfig
from wnet.backbone.u_like_net import ULikeNet
from wnet.loss.wnet_loss import WNetLoss
from wnet.data.xgzl_dataset import XGZLDataset

import os


def save_checkpoint(save_dir, ep_num, model_dict, optimizer_dict, lr_schedule_dict):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(model_dict, os.path.join(save_dir, 'model.pth'))
    torch.save(optimizer_dict, os.path.join(save_dir, 'optimizer.pth'))
    torch.save(lr_schedule_dict, os.path.join(save_dir, 'lr_schedule.pth'))
    with open(os.path.join(save_dir, 'epoch.txt'), 'w+') as f:
        f.write('{}\n'.format(ep_num))


if __name__ == '__main__':

    conf = WNetConfig()
    conf.set_cuda(True)
    conf.set_class_config(class_list=['background', 'zl'])
    conf.set_input_shape(256, 256)
    conf.set_train_info(10, 2, 1e-2)
    conf.set_eval_interval(1)
    conf.set_checkpoint_config(1, './checkpoints')
    conf.set_num_worker(0)
    conf.set_log('logs/train.log')

    logger = print if conf.log_file is None else MyNetLogger.default(conf.log_file)
    logger(conf)

    device = 'cuda' if conf.use_cuda else 'cpu'

    model = ULikeNet(conf)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), conf.init_lr, momentum=0.9, weight_decay=5e-4)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.95)

    loss_func = WNetLoss()
    loss_func.to(device)

    train_dataset = XGZLDataset('datasets/xgzl/images/training', 'datasets/xgzl/annotations/training-old', conf.input_shape)
    val_dataset = XGZLDataset('datasets/xgzl/images/validation', 'datasets/xgzl/annotations/validation-old', conf.input_shape)
    train_loader = DataLoader(train_dataset, conf.batch_size, shuffle=True, num_workers=conf.num_worker, drop_last=True)
    val_loader = DataLoader(val_dataset, conf.batch_size, shuffle=False, num_workers=conf.num_worker)

    ep_num = 0

    if conf.pretrain_path is not None:
        filename_list = os.listdir(conf.pretrain_path)
        model_filename = None
        optimizer_filename = None
        lr_schedule_filename = None
        epoch_filename =None
        for filename in filename_list:
            if filename.find('model') > -1:
                model_filename = filename
            if filename.find('optimizer') > -1:
                optimizer_filename = filename
            if filename.find('lr_schedule') > -1:
                lr_schedule_filename = filename
            if filename.find('epoch') > -1:
                epoch_filename = filename
        if model_filename is not None:
            model.load_state_dict(torch.load(os.path.join(conf.pretrain_path, model_filename)))
        if optimizer_filename is not None:
            optimizer.load_state_dict(torch.load(os.path.join(conf.pretrain_path, optimizer_filename)))
        if lr_schedule_filename is not None:
            lr_schedule.load_state_dict(torch.load(os.path.join(conf.pretrain_path, lr_schedule_filename)))
        if epoch_filename is not None:
            with open(os.path.join(conf.pretrain_path, epoch_filename), 'r') as f:
                ep_num = int(f.readlines()[0].strip())
        logger('successfully load pretrained : {}'.format(conf.pretrain_path))

    for _ in range(ep_num, conf.epoch_num):
        model.train()
        for bat_im, bat_label in train_loader:
            bat_im, bat_label = bat_im.to(device), bat_label.to(device)
            optimizer.zero_grad()
            out = model(bat_im)
            loss = loss_func(out, bat_label)
            logger('train loss: {}'.format(loss.item()))
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # break
        lr_schedule.step()

        if conf.eval_every_num is not None and ep_num % conf.eval_every_num == 0:
            model.eval()
            for bat_im, bat_label in val_loader:
                with torch.no_grad():
                    bat_im, bat_label = bat_im.to(device), bat_label.to(device)
                    out = model(bat_im)
                    loss = loss_func(out, bat_label)
                    logger('val loss: {}'.format(loss.item()))
            torch.cuda.empty_cache()
        ep_num += 1

        if conf.checkpoint_save_every_num is not None and conf.checkpoint_save_dir is not None and ep_num % conf.checkpoint_save_every_num == 0:
            save_checkpoint(
                os.path.join(conf.checkpoint_save_dir, 'epoch{}'.format(ep_num)),
                ep_num,
                model.state_dict(),
                optimizer.state_dict(),
                lr_schedule.state_dict()
            )

        # break




