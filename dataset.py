# -*- coding: UTF-8 -*-

"""
Code written by Yumeng Liang (email: liangym@bupt.edu.cn) from Beijing University of Posts and Telecommunications.
"""

import torch.utils.data as data
import numpy as np
import torch
import os
from scipy import io
from scipy.fftpack import fft
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def __len__(self):
        return 0


def read_list(list):
    image_paths = []  # data path
    kind_list = []  # liquid kind

    assert os.path.isfile(list), '%s is not a valid file' % dir
    fids = open(list, 'r')
    list_lines = fids.readlines()
    fids.close()

    for ix, line in enumerate(list_lines):
        filename = line.split(' ')[0].strip()
        image_paths.append(filename)
        kind = line.split(' ')[1].strip()
        kind_list.append(kind)

    return {'image_paths': image_paths, 'kind_list': kind_list,}


class MMWave_Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt, stage):
        self.opt = opt
        self.stage = stage
        self.root = opt.dataroot

        if (self.stage == 'train'):
            self.list_file = os.path.join(opt.dataroot, 'trainlist.txt')
        elif (self.stage == 'test'):
            self.list_file = os.path.join(opt.dataroot, 'testlist.txt')
        else:
            print('Initializing Dataset Error')
            sys.exit(1)

        list_info = read_list(self.list_file)
        self.image_paths = list_info['image_paths']
        self.data_size = len(self.image_paths)
        self.kind_list = list_info['kind_list']

    def get_frame(self, path):
        if os.path.isfile(path):
            m = io.loadmat(path)
            data = []
            data.append(m['save_mat'])
            data = torch.tensor(data, dtype=torch.float).squeeze()
            return data

    def __getitem__(self, index):

        index_i = index % self.data_size
        image_paths = self.image_paths[index_i]
        frame = self.get_frame(os.path.join(self.root, image_paths))
        kind = torch.tensor(int(self.kind_list[index_i]),dtype=torch.int)

        return {'frame': frame, 'kind': kind,}

    def __len__(self):
        return self.data_size

    def name(self):
        return 'MMWave_Dataset'
