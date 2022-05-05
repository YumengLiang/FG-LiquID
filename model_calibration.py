# -*- coding: UTF-8 -*-
"""
Code written by Yumeng Liang (email: liangym@bupt.edu.cn) from Beijing University of Posts and Telecommunications.
"""

import torch
from torch import nn


class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # i = 0
        for module in self._modules.values():
            # print(i)
            input = module(input)
            # i += 1
        return input

class MMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'MMNet'
        self.channels = [1, 16, 64, 128, 256, 256]
        self.channels_calibration = [1, 32, 64, 128, 256, 256]
        self.w_g = Sequential(
            nn.Linear(self.channels[-1], self.channels[-1] // 8),
            nn.ReLU(),
            nn.Linear(self.channels[-1] //8, 1),
            nn.ReLU(),
        )

        self.calibration = Sequential(
            nn.Linear(12 * 3, self.channels_calibration[1]),
            nn.BatchNorm1d(self.channels_calibration[1]),
            nn.ReLU(),
        )
        self.reflection = Sequential(
            nn.Linear(4*3, self.channels[1]),
            nn.BatchNorm1d(self.channels[1]),
            nn.ReLU(),
        )

        for i in range(1, len(self.channels_calibration)-1):
            self.calibration.add(nn.Linear(self.channels_calibration[i], self.channels_calibration[i+1]))
            self.calibration.add(nn.BatchNorm1d(self.channels_calibration[i+1]))
            self.calibration.add(nn.ReLU())

        for i in range(1, len(self.channels)-1):
            self.reflection.add(nn.Linear(self.channels[i], self.channels[i+1]))
            self.reflection.add(nn.BatchNorm1d(self.channels[i+1]))
            self.reflection.add(nn.ReLU())

        self.fc_kind = nn.Linear(self.channels[-1], 30)

    def forward(self, inputs):
        '''
        inputs dimension: batch_size, number of data points=3, features=4, number of Rx = 4;
        The features are [distance, elevation AoA, azimuth AoA, RSS]
        '''
        feature_calibration = inputs[:,:,:-1,:].contiguous().view(-1,12*3)
        feature_reflection = inputs[:,:,-1,:].contiguous().view(-1,12)
        #Here we divide features into location features [distance, elevation AoA, azimuth AoA] and reflection features.

        feature_calibration = self.calibration(feature_calibration)
        feature_reflection = self.reflection(feature_reflection)

        wc = self.w_g(feature_calibration)
        wr = self.w_g(feature_reflection)
        w = torch.cat([wc,wr],dim=1)
        w = torch.softmax(w,dim=1)
        wc = w[:,0:1].expand_as(feature_calibration)
        wr = w[:,1:].expand_as(feature_reflection)
        x = wc * feature_calibration + wr * feature_reflection

        kind = self.fc_kind(x)

        return kind

