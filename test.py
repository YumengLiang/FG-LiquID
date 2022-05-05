# -*- coding: UTF-8 -*-
"""
Code written by Yumeng Liang (email: liangym@bupt.edu.cn) from Beijing University of Posts and Telecommunications.
"""
import torch
import argparse
import time
from torch import nn
from model_calibration import MMNet
from dataset import MMWave_Dataset
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch Size to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    args = parser.parse_args()

    testset = MMWave_Dataset()
    testset.initialize(args, 'test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dataset_size_val = len(testloader)
    print('#test images = %d' % (dataset_size_val*args.batch_size))
    model = MMNet()
    model.load_state_dict(torch.load('model_250.pkl'))

    model.cpu()
    model.eval()

    criterion = nn.CrossEntropyLoss()

    valset = MMWave_Dataset()
    valset.initialize(args, 'test')
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)
    dataset_size_val = len(valloader)
    print('#validation images = %d' % (dataset_size_val))

    iter_data_time = time.time()

    total = 0
    correct = 0

    epoch_start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            frame = data['frame'].cpu()#.cuda()
            target = torch.tensor(data['kind'], dtype=torch.long).cpu()#.cuda()

            kind = model(frame)

            total += kind.size(0)
            _, predicted = kind.max(1)
            correct += predicted.eq(target).sum().item()

    epoch_end_time = time.time()
    print ('time: %.3f' %(epoch_end_time-epoch_start_time))
    accVal_netG_GMR = 1.0 * correct / total

    message_acc = ': %.3f (%d/%d)' % (accVal_netG_GMR, correct, total)
    print ('Accuracy_Epoch' + message_acc)
