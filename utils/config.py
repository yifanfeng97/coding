import sys
sys.path.append('../')
import os
import os.path as osp
import configparser

class config(object):
    def __init__(self, cfg_file='config/config.cfg'):
        super(config, self).__init__()
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)

        # default
        self.data_root = cfg.get('DEFAULT', 'data_root')
        self.result_root = cfg.get('DEFAULT', 'result_root')
        self.dataset = cfg.get('DEFAULT', 'dataset')
        self.data_views = cfg.getint('DEFAULT', 'data_views')
        self.input_views = cfg.getint('DEFAULT', 'input_views')
        self.val_ratio = cfg.getfloat('DEFAULT', 'val_ratio')
        self.model_type = cfg.get('DEFAULT', 'model_type')
        self.with_group = True if self.model_type=='group' else False
        self.class_num = cfg.getint('DEFAULT', 'class_num')

        # train
        self.workers = cfg.getint('TRAIN', 'workers')
        self.batch_size = cfg.getint('TRAIN', 'batch_size')
        self.lr = cfg.getfloat('TRAIN', 'lr')
        self.momentum = cfg.getfloat('TRAIN', 'momentum')
        self.weight_decay = cfg.getfloat('TRAIN', 'weight_decay')
        self.max_epoch = cfg.getint('TRAIN', 'max_epoch')
        self.print_freq = cfg.getint('TRAIN', 'print_freq')
        self.gpu_id = cfg.get('TRAIN', 'gpu_id')

        self.resume_train = cfg.getboolean('TRAIN', 'resume_train')
        self.cuda = cfg.getboolean('TRAIN', 'cuda')

        self.have_aux = cfg.getboolean('TRAIN', 'have_aux')

        self.result_sub_folder = cfg.get('TRAIN', 'result_sub_folder')
        self.ckpt_folder = cfg.get('TRAIN', 'ckpt_folder')
        self.split_folder = cfg.get('TRAIN', 'split_folder')

        self.split_train = cfg.get('TRAIN', 'split_train')
        self.split_test = cfg.get('TRAIN', 'split_test')
        self.ckpt_model = cfg.get('TRAIN', 'ckpt_model')
        self.ckpt_optim = cfg.get('TRAIN', 'ckpt_optim')

        self.check_dirs()


    def check_dirs(self):
        self.check_dir(self.result_root)
        self.check_dir(self.result_sub_folder)
        self.check_dir(self.ckpt_folder)
        self.check_dir(self.split_folder)

    def check_dir(self, _dir):
        if not osp.exists(_dir):
            os.mkdir(_dir)

