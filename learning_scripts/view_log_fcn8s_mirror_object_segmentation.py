#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import os.path as osp


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
    train_loss_mirror = []
    train_loss_object = []
    train_mean_iu_mirror = []
    train_mean_iu_object = []

    val_loss = []
    val_loss_mirror = []
    val_loss_object = []
    val_mean_iu_mirror = []
    val_mean_iu_object = []

    for tmp in data:
        if 'iteration' in tmp:
            iteration.append(tmp['iteration'])
        if 'epoch' in tmp and 'validation/main/loss' in tmp:
            epoch.append(tmp['epoch'])

        if 'main/loss' in tmp:
            train_loss.append(tmp['main/loss'])
        if 'main/loss_mirror' in tmp:
            train_loss_mirror.append(tmp['main/loss_mirror'])
        if 'main/loss_object' in tmp:
            train_loss_object.append(tmp['main/loss_object'])
        if 'main/mean_iu_mirror' in tmp:
            train_mean_iu_mirror.append(tmp['main/mean_iu_mirror'])
        if 'main/mean_iu_object' in tmp:
            train_mean_iu_object.append(tmp['main/mean_iu_object'])

        if 'epoch' in tmp and 'validation/main/loss' in tmp:
            val_loss.append(tmp['validation/main/loss'])
        if 'epoch' in tmp and 'validation/main/loss_mirror' in tmp:
            val_loss_mirror.append(tmp['validation/main/loss_mirror'])
        if 'epoch' in tmp and 'validation/main/loss_object' in tmp:
            val_loss_object.append(tmp['validation/main/loss_object'])
        if 'epoch' in tmp and 'validation/main/mean_iu_mirror' in tmp:
            val_mean_iu_mirror.append(tmp['validation/main/mean_iu_mirror'])
        if 'epoch' in tmp and 'validation/main/mean_iu_object' in tmp:
            val_mean_iu_object.append(tmp['validation/main/mean_iu_object'])

    do_not_show_before = 0  # please fill [iteration / 20]
    iteration = iteration[do_not_show_before:]
    train_loss = train_loss[do_not_show_before:]
    train_loss_mirror = train_loss_mirror[do_not_show_before:]
    train_loss_object = train_loss_object[do_not_show_before:]
    train_mean_iu_mirror = train_mean_iu_mirror[do_not_show_before:]
    train_mean_iu_object = train_mean_iu_object[do_not_show_before:]

    fig = plt.figure(figsize=(16, 9))
    fig01 = plt.subplot2grid((2, 5), (0, 0))
    fig02 = plt.subplot2grid((2, 5), (0, 1))
    fig03 = plt.subplot2grid((2, 5), (0, 2))
    fig04 = plt.subplot2grid((2, 5), (0, 3))
    fig05 = plt.subplot2grid((2, 5), (0, 4))
    fig06 = plt.subplot2grid((2, 5), (1, 0))
    fig07 = plt.subplot2grid((2, 5), (1, 1))
    fig08 = plt.subplot2grid((2, 5), (1, 2))
    fig09 = plt.subplot2grid((2, 5), (1, 3))
    fig10 = plt.subplot2grid((2, 5), (1, 4))

    fig01.plot(iteration, train_loss)
    fig01.set_title('main/loss')
    fig01.set_xlabel('iteration [-]')
    fig01.set_ylabel('loss [-]')
    fig01.grid(True)

    fig02.plot(iteration, train_loss_mirror)
    fig02.set_title('main/loss_mirror')
    fig02.set_xlabel('iteration [-]')
    fig02.set_ylabel('loss [-]')
    fig02.grid(True)

    fig03.plot(iteration, train_loss_object)
    fig03.set_title('main/loss_object')
    fig03.set_xlabel('iteration [-]')
    fig03.set_ylabel('loss [-]')
    fig03.grid(True)

    fig04.plot(iteration, train_mean_iu_mirror)
    fig04.set_title('main/mean_iu_mirror')
    fig04.set_xlabel('iteration [-]')
    fig04.set_ylabel('Mean IU [-]')
    fig04.grid(True)

    fig05.plot(iteration, train_mean_iu_object)
    fig05.set_title('main/mean_iu_object')
    fig05.set_xlabel('iteration [-]')
    fig05.set_xlabel('Mean IU [-]')
    fig05.grid(True)

    fig06.plot(epoch, val_loss)
    fig06.set_title('validation/main/loss')
    fig06.set_xlabel('epoch')
    fig06.grid(True)

    fig07.plot(epoch, val_loss_mirror)
    fig07.set_title('validation/main/loss_mirror')
    fig07.set_xlabel('epoch')
    fig07.grid(True)

    fig08.plot(epoch, val_loss_object)
    fig08.set_title('validation/main/loss_object')
    fig08.set_xlabel('epoch')
    fig08.grid(True)

    fig09.plot(epoch, val_mean_iu_mirror)
    fig09.set_title('validation/main/mean_iu_mirror')
    fig09.set_xlabel('epoch')
    fig09.grid(True)

    fig10.plot(epoch, val_mean_iu_object)
    fig10.set_title('validation/main/mean_iu_object')
    fig10.set_xlabel('epoch')
    fig10.grid(True)

    fig.tight_layout()
    fig.show()

    plt.waitforbuttonpress(0)
    plt.close(fig)
    return


if __name__ == '__main__':
    main()
