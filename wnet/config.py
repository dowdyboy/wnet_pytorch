

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
        self.log_dir = None

        # loader的worker个数
        self.num_worker = 0

