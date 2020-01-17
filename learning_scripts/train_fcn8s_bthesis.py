#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
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
import cv2
import fcn
import numpy as np

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from datasets import MultiViewMirror3DAnnotatedDataset  # NOQA
from datasets import TransparentObjects3DAnnotatedDataset  # NOQA
from models import FCN8sAtOnceConcatAtOnce  # NOQA
from models import FCN8sAtOnceInputRGBD  # NOQA


here = osp.dirname(osp.abspath(__file__))


def colorize_depth(depth, min_value=None, max_value=None):
    min_value = np.nanmin(depth) if min_value is None else min_value
    max_value = np.nanmax(depth) if max_value is None else max_value

    gray_depth = depth.copy()
    nan_mask = np.isnan(gray_depth)
    gray_depth[nan_mask] = 0
    gray_depth = 255 * (gray_depth - min_value) / (max_value - min_value)
    gray_depth[gray_depth < 0] = 0
    gray_depth[gray_depth > 255] = 255
    gray_depth = gray_depth.astype(np.uint8)
    colorized = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
    colorized[nan_mask] = (0, 0, 0)

    return colorized


def transform(in_data):
    # min_value = 0.5
    # max_value = 5.0

    label_gt = in_data[0][2]
    depth_gt = in_data[0][3]

    image_bgrs = None
    depth_nan2zeros = None
    # depth_bgrs = None

    for example in in_data:
        image_rgb, depth, label_gt, depth_gt, _ = example

        # RGB -> BGR
        image_bgr = image_rgb[:, :, ::-1]
        image_bgr = image_rgb.astype(np.float32)
        mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        image_bgr -= mean_bgr
        # (H, W, 3) -> (3, H, W)
        image_bgr = image_bgr.transpose((2, 0, 1))

        # depth -> depth_nan2zero: (H, W) -> (1, H, W) -> (3, H, W)
        depth_nan2zero = depth.copy()
        depth_nan2zero[np.isnan(depth_nan2zero)] = 0.0
        depth_nan2zero = depth_nan2zero[np.newaxis, :, :]
        depth_nan2zero_3ch = np.concatenate(
            (depth_nan2zero, depth_nan2zero), axis=0)
        depth_nan2zero_3ch = np.concatenate(
            (depth_nan2zero_3ch, depth_nan2zero), axis=0)

        # depth -> depth_bgr: (H, W) -> (H, W, 3) -> (3, H, W)
        # depth_bgr = colorize_depth(
        #     depth, min_value=min_value, max_value=max_value)
        # depth_bgr = depth_bgr.astype(np.float32)
        # depth_bgr -= mean_bgr
        # depth_bgr = depth_bgr.transpose((2, 0, 1))

        # Append to list
        if image_bgrs is None:
            image_bgrs = image_bgr
            depth_nan2zeros = depth_nan2zero
            # depth_bgrs = depth_bgr
        else:
            # (3, H, W) -> (3 * num_view, H, W)
            image_bgrs = np.concatenate((image_bgrs, image_bgr), axis=0)
            # (1, H, W) -> (1 * num_view, H, W)
            depth_nan2zeros = np.concatenate(
                (depth_nan2zeros, depth_nan2zero), axis=0)
            # (3, H, W) -> (3 * num_view, H, W)
            # depth_bgrs = np.concatenate((depth_bgrs, depth_bgr), axis=0)

    # return image_bgrs, depth_nan2zeros, depth_bgrs, label_gt, depth_gt
    # return image_bgrs, depth_bgrs, label_gt, depth_gt
    return image_bgrs, depth_nan2zero_3ch, label_gt, depth_gt


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-g', '--gpu', default=0, type=int, help='GPU id')
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, help='Dataset class name')
    parser.add_argument(
        '-m', '--model', type=str, required=True, help='Model class name')
    parser.add_argument(
        '-n', '--num_view', type=int, required=True,
        help='Number of getting examples at the same time')
    parser.add_argument(
        '-b', '--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument(
        '-e', '--epoch', type=int, required=True, help='Training epoch')
    args = parser.parse_args()

    gpu = args.gpu
    num_view = args.num_view

    # 0. config

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = timestamp
    out = osp.join(osp.dirname(here), 'logs', out)

    max_iter_epoch = args.epoch, 'epoch'
    progress_bar_update_interval = 10  # iteration
    print_interval = 100, 'iteration'
    log_interval = 100, 'iteration'
    test_interval = 5, 'epoch'
    save_interval = 5, 'epoch'

    # 1. dataset

    if args.dataset == 'MultiViewMirror3DAnnotatedDataset':
        dataset_train = MultiViewMirror3DAnnotatedDataset(
            split='train', aug=True, num_view=num_view)
        dataset_valid = MultiViewMirror3DAnnotatedDataset(
            split='test', aug=False, num_view=num_view)
    elif args.dataset == 'TransparentObjects3DAnnotatedDataset':
        dataset_train = TransparentObjects3DAnnotatedDataset(
            split='train', aug=True, num_view=num_view)
        dataset_valid = TransparentObjects3DAnnotatedDataset(
            split='test', aug=False, num_view=num_view)
    else:
        print('Invalid dataset class.')
        exit(1)

    dataset_train_transformed = TransformDataset(dataset_train, transform)
    dataset_valid_transformed = TransformDataset(dataset_valid, transform)

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train_transformed, batch_size=args.batch_size,
        shared_mem=10 ** 8)
    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset_valid_transformed, batch_size=1, shared_mem=10 ** 8,
        repeat=False, shuffle=False)

    # 2. model

    vgg = fcn.models.VGG16()
    vgg_path = vgg.download()
    chainer.serializers.load_npz(vgg_path, vgg)

    n_class = len(dataset_train.class_names)
    assert n_class == 2

    if args.model == 'FCN8sAtOnceInputRGBD':
        model = FCN8sAtOnceInputRGBD(
            n_class=n_class, masking=True, concat=True,
            no_bp_before_rgb_pool5=False)
    elif args.model == 'FCN8sAtOnceConcatAtOnce':
        model = FCN8sAtOnceConcatAtOnce(
            n_class=n_class, masking=True, no_bp_before_rgb_pool5=False)
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

    trainer = chainer.training.Trainer(updater, max_iter_epoch, out=out)

    trainer.extend(extensions.ExponentialShift("alpha", 0.99997))

    if not osp.isdir(out):
        os.makedirs(out)

    with open(osp.join(out, 'dataset.txt'), 'w') as f:
        f.write(dataset_train.__class__.__name__)

    with open(osp.join(out, 'model.txt'), 'w') as f:
        f.write(model.__class__.__name__)

    with open(osp.join(out, 'n_class.txt'), 'w') as f:
        f.write(str(n_class))

    with open(osp.join(out, 'batch_size.txt'), 'w') as f:
        f.write(str(args.batch_size))

    with open(osp.join(out, 'num_view.txt'), 'w') as f:
        f.write(str(num_view))

    trainer.extend(
        extensions.snapshot_object(
            model,
            savefun=chainer.serializers.save_npz,
            filename='max_miou.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            'validation/main/miou', save_interval))
    trainer.extend(
        extensions.snapshot_object(
            model,
            savefun=chainer.serializers.save_npz,
            filename='max_depth_acc.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            'validation/main/depth_acc<0.10', save_interval))

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
        'main/seg_loss',
        'main/reg_loss',
        'main/miou',
        'main/depth_acc<0.03',
        'main/depth_acc<0.10',
        'main/depth_acc<0.30',
        'validation/main/miou',
        'validation/main/depth_acc<0.03',
        'validation/main/depth_acc<0.10',
        'validation/main/depth_acc<0.30',
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
