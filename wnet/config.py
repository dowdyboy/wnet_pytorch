import os


# 配置封装
class WNetConfig:

    def __init__(self):
        super(WNetConfig, self).__init__()
        # 是否使用cuda
        self.use_cuda = False
        # 分类名称列表
        self.class_list = []
        # 分类个数
        self.num_class = 0

        # 预训练路径
        self.pretrain_path = None

        # 网络输入大小
        self.input_shape = None

        # 训练epoch个数
        self.epoch_num = 0
        # batch size
        self.batch_size = 0
        # 初始学习率
        self.init_lr = 0.

        # 每多少次记录一个检查点
        self.checkpoint_save_every_num = None
        # 检查点存储目录
        self.checkpoint_save_dir = None
        # 每多少次进行一次评估
        self.eval_every_num = None

        # 日志目录
        self.log_file = None

        # loader的worker个数
        self.num_worker = 0

    # 设置类别相关信息
    def set_class_config(self, path=None, class_list=None):
        if class_list is not None:
            self.class_list = class_list
            self.num_class = len(class_list)
        elif path is not None:
            with open(path, 'r') as f:
                lines = f.readlines()
                self.class_list = [c.strip() for c in lines]
                self.num_class = len(self.class_list)

    # 设置网络输入大小
    def set_input_shape(self, w, h):
        if w % 32 != 0 or h % 32 != 0:
            raise ValueError('input shape must % 32')
        self.input_shape = (h, w)

    # 设置训练相关信息
    def set_train_info(self, epoch_num, batch_size, lr):
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.init_lr = lr

    # 设置是否使用cuda
    def set_cuda(self, use_cuda):
        self.use_cuda = use_cuda

    # 设置loader的worker个数
    def set_num_worker(self, num):
        self.num_worker = num

    # 设置检查点配置信息
    def set_checkpoint_config(self, checkpoint_save_every_num, checkpoint_save_dir):
        self.checkpoint_save_every_num = checkpoint_save_every_num
        self.checkpoint_save_dir = checkpoint_save_dir
        if not os.path.isdir(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)

    # 设置训练时多少次进行一次评估
    def set_eval_interval(self, eval_every_num):
        self.eval_every_num = eval_every_num

    # 设置预训练状态路径
    def set_pretrained_path(self, pretrain_path):
        self.pretrain_path = pretrain_path

    # 配置日志信息
    def set_log(self, log_file):
        self.log_file = log_file
        if not os.path.isdir(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))

    def __str__(self):
        ret = '[WNET CONFIG]\n'
        attrs = list(filter(lambda x: not str(x).startswith('__') and not str(x).startswith('set'), dir(self)))
        sorted(attrs)
        for a in attrs:
            ret += '{}: {}\n'.format(a, getattr(self, a))
        return ret