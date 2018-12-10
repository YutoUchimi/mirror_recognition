#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp
import sys

import chainer
from chainer import cuda
from chainer.datasets import TransformDataset
from chainer.training import extensions
import fcn
import numpy as np

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from datasets import LabeledMirrorDataset  # NOQA
from models import FCN8sMirrorSegmentation  # NOQA


here = osp.dirname(osp.abspath(__file__))


def transform(in_data):
    rgb_img, gt = in_data

    # RGB -> BGR
    bgr_img = rgb_img[:, :, ::-1]
    bgr_img = rgb_img.astype(np.float32)
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    bgr_img -= mean_bgr
    # H, W, C -> C, H, W
    bgr_img = bgr_img.transpose((2, 0, 1))

    return bgr_img, gt


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-g', '--gpu', default=0, type=int, help='GPU id')
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, help='Dataset class name')
    parser.add_argument(
        '-m', '--model', type=str, required=True, help='Model class name')
    args = parser.parse_args()

    gpu = args.gpu

    # 0. config

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = timestamp
    out = osp.join(osp.dirname(here), 'logs', out)

    max_iteration = 10000
    progress_bar_update_interval = 10  # iteration
    print_interval = 100, 'iteration'
    log_interval = 100, 'iteration'
    test_interval = 20, 'epoch'
    save_interval = 20, 'epoch'

    # 1. dataset

    if args.dataset == 'LabeledMirrorDataset':
        dataset_train = LabeledMirrorDataset(split='train', aug=True)
        dataset_valid = LabeledMirrorDataset(split='test', aug=False)
    else:
        print('Invalid dataset class.')
        exit(1)

    dataset_train_transformed = TransformDataset(dataset_train, transform)
    dataset_valid_transformed = TransformDataset(dataset_valid, transform)

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train_transformed, batch_size=1, shared_mem=10 ** 7)
    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset_valid_transformed, batch_size=1, shared_mem=10 ** 7,
        repeat=False, shuffle=False)

    # 2. model

    vgg = fcn.models.VGG16()
    vgg_path = vgg.download()
    chainer.serializers.load_npz(vgg_path, vgg)

    n_class = len(dataset_train.class_names)

    if args.model == 'FCN8sMirrorSegmentation':
        model = FCN8sMirrorSegmentation(n_class=n_class)
    else:
        print('Invalid model class.')
        exit(1)

    model.init_from_vgg16(vgg)

    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.Adam(alpha=1.0e-5)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    updater = chainer.training.updater.StandardUpdater(
        iter_train, optimizer, device=gpu)

    trainer = chainer.training.Trainer(
        updater, (max_iteration, 'iteration'), out=out)

    trainer.extend(extensions.ExponentialShift("alpha", 0.99999))

    if not osp.isdir(out):
        os.makedirs(out)

    with open(osp.join(out, 'dataset.txt'), 'w') as f:
        f.write(dataset_train.__class__.__name__)

    with open(osp.join(out, 'model.txt'), 'w') as f:
        f.write(model.__class__.__name__)

    with open(osp.join(out, 'n_class.txt'), 'w') as f:
        f.write(str(n_class))

    # trainer.extend(
    #     extensions.snapshot_object(
    #         model,
    #         savefun=chainer.serializers.save_npz,
    #         filename='iter_{.updater.iteration}.npz'),
    #     trigger=save_interval)
    trainer.extend(
        extensions.snapshot_object(
            model,
            savefun=chainer.serializers.save_npz,
            filename='max_miou.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            'validation/main/miou', save_interval))

    trainer.extend(
        extensions.dump_graph(
            root_name='main/loss',
            out_name='graph.dot'))

    trainer.extend(
        extensions.LogReport(
            log_name='log.json',
            trigger=log_interval))

    trainer.extend(chainer.training.extensions.PrintReport([
        'iteration',
        'epoch',
        'elapsed_time',
        'lr',
        'main/loss',
        'main/miou',
        'validation/main/miou',
    ]), trigger=print_interval)

    trainer.extend(
        extensions.observe_lr(),
        trigger=log_interval)
    trainer.extend(
        extensions.ProgressBar(update_interval=progress_bar_update_interval))
    trainer.extend(
        extensions.Evaluator(iter_valid, model, device=gpu),
        trigger=test_interval)

    trainer.run()


if __name__ == '__main__':
    main()
