#!/usr/bin/env python

import argparse
import json
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timestamp', required=True,
                        help='Directory name to view log.')
    args = parser.parse_args()
    timestamp = args.timestamp

    here = osp.dirname(osp.abspath(__file__))
    logs_dir = osp.join(osp.dirname(here), 'logs')
    log_dir = osp.join(logs_dir, timestamp)

    with open(osp.join(log_dir, 'log.json')) as f:
        data = json.load(f)

    iteration = []
    epoch = []

    train_loss = []
    train_seg_loss = []
    train_reg_loss = []
    train_miou = []
    train_depth_acc_001 = []
    train_depth_acc_003 = []
    train_depth_acc_010 = []
    train_depth_acc_030 = []
    train_depth_acc_100 = []

    val_miou = []
    val_depth_acc_001 = []
    val_depth_acc_003 = []
    val_depth_acc_010 = []
    val_depth_acc_030 = []
    val_depth_acc_100 = []

    for tmp in data:
        if 'iteration' in tmp:
            iteration.append(tmp['iteration'])
        if 'epoch' in tmp and 'validation/main/loss' in tmp:
            epoch.append(tmp['epoch'])

        if 'main/loss' in tmp:
            train_loss.append(tmp['main/loss'])
        if 'main/seg_loss' in tmp:
            train_seg_loss.append(tmp['main/seg_loss'])
        if 'main/reg_loss' in tmp:
            train_reg_loss.append(tmp['main/reg_loss'])
        if 'main/miou' in tmp:
            train_miou.append(tmp['main/miou'])
        if 'main/depth_acc<0.01' in tmp:
            if not np.isnan(tmp['main/depth_acc<0.01']):
                train_depth_acc_001.append(tmp['main/depth_acc<0.01'])
            elif len(train_depth_acc_001) > 0:
                train_depth_acc_001.append(train_depth_acc_001[-1])
            else:
                train_depth_acc_001.append(0.0)
        if 'main/depth_acc<0.03' in tmp:
            if not np.isnan(tmp['main/depth_acc<0.03']):
                train_depth_acc_003.append(tmp['main/depth_acc<0.03'])
            elif len(train_depth_acc_003) > 0:
                train_depth_acc_003.append(train_depth_acc_003[-1])
            else:
                train_depth_acc_003.append(0.0)
        if 'main/depth_acc<0.10' in tmp:
            if not np.isnan(tmp['main/depth_acc<0.10']):
                train_depth_acc_010.append(tmp['main/depth_acc<0.10'])
            elif len(train_depth_acc_010) > 0:
                train_depth_acc_010.append(train_depth_acc_010[-1])
            else:
                train_depth_acc_010.append(0.0)
        if 'main/depth_acc<0.30' in tmp:
            if not np.isnan(tmp['main/depth_acc<0.30']):
                train_depth_acc_030.append(tmp['main/depth_acc<0.30'])
            elif len(train_depth_acc_030) > 0:
                train_depth_acc_030.append(train_depth_acc_030[-1])
            else:
                train_depth_acc_030.append(0.0)
        if 'main/depth_acc<1.00' in tmp:
            if not np.isnan(tmp['main/depth_acc<1.00']):
                train_depth_acc_100.append(tmp['main/depth_acc<1.00'])
            elif len(train_depth_acc_100) > 0:
                train_depth_acc_100.append(train_depth_acc_100[-1])
            else:
                train_depth_acc_100.append(0.0)

        if 'epoch' in tmp and 'validation/main/miou' in tmp:
            val_miou.append(tmp['validation/main/miou'])
        if 'epoch' in tmp and 'validation/main/depth_acc<0.01' in tmp:
            val_depth_acc_001.append(tmp['validation/main/depth_acc<0.01'])
        if 'epoch' in tmp and 'validation/main/depth_acc<0.03' in tmp:
            val_depth_acc_003.append(tmp['validation/main/depth_acc<0.03'])
        if 'epoch' in tmp and 'validation/main/depth_acc<0.10' in tmp:
            val_depth_acc_010.append(tmp['validation/main/depth_acc<0.10'])
        if 'epoch' in tmp and 'validation/main/depth_acc<0.30' in tmp:
            val_depth_acc_030.append(tmp['validation/main/depth_acc<0.30'])
        if 'epoch' in tmp and 'validation/main/depth_acc<1.00' in tmp:
            val_depth_acc_100.append(tmp['validation/main/depth_acc<1.00'])

    # do_not_show_before = 0  # please fill [iteration / 20]
    # iteration = iteration[do_not_show_before:]
    # train_loss = train_loss[do_not_show_before:]
    # train_miou = train_miou[do_not_show_before:]

    fig = plt.figure(figsize=(20, 6))
    fig00 = plt.subplot2grid((2, 5), (0, 0))
    fig01 = plt.subplot2grid((2, 5), (0, 1))
    fig02 = plt.subplot2grid((2, 5), (0, 2))
    fig03 = plt.subplot2grid((2, 5), (0, 3))
    fig04 = plt.subplot2grid((2, 5), (0, 4))
    fig13 = plt.subplot2grid((2, 5), (1, 3))
    fig14 = plt.subplot2grid((2, 5), (1, 4))

    fig00.plot(iteration, train_loss)
    fig00.set_title('main/loss')
    fig00.set_xlabel('iteration [-]')
    fig00.set_ylabel('loss [-]')
    fig00.grid(True)

    fig01.plot(iteration, train_seg_loss)
    fig01.set_title('main/seg_loss')
    fig01.set_xlabel('iteration [-]')
    fig01.set_ylabel('loss [-]')
    fig01.grid(True)

    fig02.plot(iteration, train_reg_loss)
    fig02.set_title('main/reg_loss')
    fig02.set_xlabel('iteration [-]')
    fig02.set_ylabel('loss [-]')
    fig02.grid(True)

    fig03.plot(iteration, train_miou)
    fig03.set_title('main/miou')
    fig03.set_xlabel('iteration [-]')
    fig03.set_ylabel('Mean IoU [-]')
    fig03.grid(True)

    fig04.plot(iteration, train_depth_acc_001)
    fig04.plot(iteration, train_depth_acc_003)
    fig04.plot(iteration, train_depth_acc_010)
    fig04.plot(iteration, train_depth_acc_030)
    fig04.plot(iteration, train_depth_acc_100)
    fig04.set_title('main/depth_acc<?.??')
    fig04.set_xlabel('iteration [-]')
    fig04.set_ylabel('depth accuracy [-]')
    fig04.grid(True)

    fig13.plot(epoch, val_miou)
    fig13.set_title('validation/main/miou')
    fig13.set_xlabel('epoch [-]')
    fig13.set_ylabel('Mean IoU [-]')
    fig13.grid(True)

    fig14.plot(epoch, val_depth_acc_001)
    fig14.plot(epoch, val_depth_acc_003)
    fig14.plot(epoch, val_depth_acc_010)
    fig14.plot(epoch, val_depth_acc_030)
    fig14.plot(epoch, val_depth_acc_100)
    fig14.set_title('validation/main/depth_acc<?.??')
    fig14.set_xlabel('epoch [-]')
    fig14.set_ylabel('depth accuracy [-]')
    fig14.grid(True)

    fig.tight_layout()
    fig.show()

    plt.waitforbuttonpress(0)
    plt.close(fig)
    return


if __name__ == '__main__':
    main()