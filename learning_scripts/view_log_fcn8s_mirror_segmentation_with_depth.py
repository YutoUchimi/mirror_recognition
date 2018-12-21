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
    train_miou = []

    val_loss = []
    val_miou = []

    for tmp in data:
        if 'iteration' in tmp:
            iteration.append(tmp['iteration'])
        if 'epoch' in tmp and 'validation/main/loss' in tmp:
            epoch.append(tmp['epoch'])

        if 'main/loss' in tmp:
            train_loss.append(tmp['main/loss'])
        if 'main/miou' in tmp:
            train_miou.append(tmp['main/miou'])

        if 'epoch' in tmp and 'validation/main/loss' in tmp:
            val_loss.append(tmp['validation/main/loss'])
        if 'epoch' in tmp and 'validation/main/miou' in tmp:
            val_miou.append(tmp['validation/main/miou'])

    do_not_show_before = 0  # please fill [iteration / 20]
    iteration = iteration[do_not_show_before:]
    train_loss = train_loss[do_not_show_before:]
    train_miou = train_miou[do_not_show_before:]

    fig = plt.figure(figsize=(8, 6))
    fig00 = plt.subplot2grid((2, 2), (0, 0))
    fig01 = plt.subplot2grid((2, 2), (0, 1))
    fig10 = plt.subplot2grid((2, 2), (1, 0))
    fig11 = plt.subplot2grid((2, 2), (1, 1))

    fig00.plot(iteration, train_loss)
    fig00.set_title('main/loss')
    fig00.set_xlabel('iteration [-]')
    fig00.set_ylabel('loss [-]')
    fig00.grid(True)

    fig01.plot(iteration, train_miou)
    fig01.set_title('main/miou')
    fig01.set_xlabel('iteration [-]')
    fig01.set_xlabel('Mean IU [-]')
    fig01.grid(True)

    fig10.plot(epoch, val_loss)
    fig10.set_title('validation/main/loss')
    fig10.set_xlabel('epoch')
    fig10.grid(True)

    fig11.plot(epoch, val_miou)
    fig11.set_title('validation/main/miou')
    fig11.set_xlabel('epoch')
    fig11.grid(True)

    fig.tight_layout()
    fig.show()

    plt.waitforbuttonpress(0)
    plt.close(fig)
    return


if __name__ == '__main__':
    main()
