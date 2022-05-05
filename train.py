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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=480,
                        help='Batch Size to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--save_prediction', type=bool, default=False,
                        help='Flags to decide whether to save the prediction.')
    args = parser.parse_args()

    # load training set
    trainset = MMWave_Dataset()
    trainset.initialize(args, 'train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dataset_size = len(trainloader)
    print('#training images = %d' % (dataset_size*args.batch_size))

    # load test set
    testset = MMWave_Dataset()
    testset.initialize(args, 'test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dataset_size_val = len(testloader)
    print('#test images = %d' % (dataset_size_val*args.batch_size))

    # build the network
    model = MMNet()
    model.cuda()
    model.train()

    total_steps = 0
    best_eval_kind = 0    # variable to record highest accuracy, initialized to 0
    best_epoch=0          # variable to record epoch with highest accuracy, initialized to 0

    # choose the lose and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer =  torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # build a dataset loader with batch size of 1
    if args.save_prediction:
        valset = MMWave_Dataset()
        valset.initialize(args, 'test')
        valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)
        dataset_size_val = len(valloader)
        print('#validation images = %d' % (dataset_size_val))

    train_start_time = time.time()
    for epoch in range(0, args.epochs+1):

        epoch_iter = 0
        train_loss = 0      # variable to count the training loss, initialized to 0
        total = 0
        correct = 0         # variable to count the correct prediction, initialized to 0
        with open('log.txt', "a") as log_file:
            for i, data in enumerate(trainloader):
                iter_start_time = time.time()

                total_steps += args.batch_size
                epoch_iter += args.batch_size
                optimizer.zero_grad()

                # input data
                frame = data['frame'].cuda()
                kind_target = torch.tensor(data['kind'],dtype=torch.long).cuda()

                # forward and loss calculation
                kind = model(frame)
                loss = criterion(kind, kind_target)

                #regularization
                lmbd = 0.9* 1e-4
                reg_loss =None
                for param in model.parameters():
                    if reg_loss is None:
                        reg_loss = 0.5 * torch.sum(param ** 2)
                    else:
                        reg_loss = reg_loss + 0.5 * param.norm(2) ** 2
                loss += lmbd * reg_loss

                # backward
                loss.backward()
                train_loss += loss.item()

                # prediction count
                _, predicted = kind.max(1)
                total += kind.size(0)
                correct += predicted.eq(kind_target).sum().item()
                optimizer.step()

                # compute running time
                t = (time.time() - iter_start_time) / args.batch_size

            if (epoch %50==0):
                msg = 'epoch: %d, epoch_iter: %d, Loss: %.3f | Acc: %.3f%% (%d/%d) | time: %.3f ms' % (
                    epoch, epoch_iter, train_loss, 100. * correct / total, correct, total, t * 1000.0)
                print(msg)

                for param_group in optimizer.param_groups:
                    print ('lr: ', param_group['lr'])
                log_file.write(msg)
        scheduler.step()
        model.eval()

        total = 0
        correct = 0
        test_loss = 0

        # test on the test set and print the accuracy
        with torch.no_grad():
            for i, data in enumerate(testloader):
                frame = data['frame'].cuda()
                target = torch.tensor(data['kind'], dtype=torch.long).cuda()

                kind = model(frame)
                loss = criterion(kind, target)
                test_loss += loss.item()

                total += kind.size(0)
                _, predicted = kind.max(1)
                correct += predicted.eq(target).sum().item()

        accVal_netG_GMR = 1.0 * correct / total

        if accVal_netG_GMR > best_eval_kind :
            best_eval_kind = accVal_netG_GMR
            best_epoch = epoch

            # save the prediction
            if args.save_prediction:
                model.eval()
                with torch.no_grad():
                    with open('result.txt', "w") as f:
                        for i, data in enumerate(valloader):
                            frame = data['frame'].cuda()
                            kind = model(frame)
                            _, predicted_k = kind.max(1)

                            f.write("pre %d " % (predicted_k))
                            f.write("gt %d\n" % (data['kind']))

        if (epoch % 50 == 0):
            print ('----EVAL---- Loss: %.3f' % test_loss)
            message_acc = 'Accuracy_Epoch (epoch: %d ) ' % (epoch)
            message_acc += ': %.3f (%d/%d)' % (accVal_netG_GMR,correct,total)

            with open('log.txt', "a") as log_file:
                log_file.write('%s\n' % message_acc)

                message_acc += 'best epoch: %d, %.3f' % (best_epoch, best_eval_kind)
                print (message_acc)
                log_file.write('%s\n' % message_acc)
                if (best_eval_kind>0.9):
                    PATH = './'+'model_%03d'%(epoch)+'.pkl'
                    torch.save(model.state_dict(), PATH)
        model.train()

    train_end_time = time.time()

    print (train_end_time-train_start_time)