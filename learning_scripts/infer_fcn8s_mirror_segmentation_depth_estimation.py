#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from distutils.version import LooseVersion
import os.path as osp
import sys

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.serializers as S
import cv2
import fcn
import mvtk
import numpy as np

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from datasets import Mirror3DAnnotatedDataset  # NOQA
from models import FCN8sMirrorSegmentationDepthEstimation  # NOQA


class FCNMirrorSegmentationDepthEstimation(object):

    class_names = np.array([
        '_background_',
        'mirror',
    ], dtype=np.str)
    class_names.setflags(write=0)

    min_value = 0.5
    max_value = 5.0

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--gpu', default=0, type=int,
                            help='GPU id. -1 is CPU mode.')
        parser.add_argument('-t', '--timestamp', required=True,
                            help='Directory name to load model')
        parser.add_argument('-s', '--split', required=True,
                            choices=['train', 'test'])
        parser.add_argument('-S', '--summarize', action='store_true',
                            help='Enable to summarize depth accuracy.')
        args = parser.parse_args()

        self.split = args.split
        self.gpu = args.gpu
        self.summarize = args.summarize
        if osp.isdir(args.timestamp):
            self.data_dir = args.timestamp
        else:
            full_path = osp.join(
                osp.dirname(osp.dirname(osp.abspath(__file__))),
                'logs',
                args.timestamp)
            if osp.isdir(full_path):
                self.data_dir = full_path
            else:
                print('Invalid timestamp.')
                exit(1)

        self.bg_label = 0
        self.proba_threshold = 0.5

        self.load_model()
        self.load_dataset()
        if self.summarize is True:
            self.summarizing_process()
        self.process()

    def load_model(self):
        model_file = osp.join(self.data_dir, 'max_miou.npz')
        with open(osp.join(self.data_dir, 'model.txt'), 'r') as f:
            model_name = f.readline().rstrip()
        with open(osp.join(self.data_dir, 'n_class.txt'), 'r') as f:
            n_class = int(f.readline().rstrip())
        self.n_class = n_class

        if model_name == 'FCN8sMirrorSegmentationDepthEstimation':
            self.model = FCN8sMirrorSegmentationDepthEstimation(
                n_class=n_class)

        print('\nLoading trained model:          {0}'.format(model_file))
        S.load_npz(model_file, self.model)
        print('Finished loading trained model: {0}\n'.format(model_file))

        if self.gpu >= 0:
            cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
            self.model.train = False

    def load_dataset(self):
        with open(osp.join(self.data_dir, 'dataset.txt'), 'r') as f:
            dataset_name = f.readline().rstrip()
        if dataset_name == 'Mirror3DAnnotatedDataset':
            self.dataset = Mirror3DAnnotatedDataset(
                split=self.split, aug=False)
        self.data_len = len(self.dataset)
        print('dataset class name: %s' % self.dataset.__class__.__name__)
        print('split: %s' % self.split)
        print('dataset length: %d' % self.data_len)
        print('n_class: %d' % self.n_class)

    def _colorize_depth(self, depth, min_value=None, max_value=None):
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

    def _transform(self, image_rgb, depth, label_gt, depth_gt):
        # RGB -> BGR
        image_bgr = image_rgb[:, :, ::-1]
        image_bgr = image_bgr.astype(np.float32)
        mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        image_bgr -= mean_bgr
        # H, W, C -> C, H, W
        image_bgr = image_bgr.transpose((2, 0, 1))

        # depth -> depth_nan2zero: (H, W) -> (1, H, W)
        depth_nan2zero = depth.copy()
        depth_nan2zero[np.isnan(depth_nan2zero)] = 0.0
        depth_nan2zero = depth_nan2zero[np.newaxis, :, :]

        # depth (H, W) -> (H, W, 3) -> (3, H, W)
        depth_bgr = self._colorize_depth(
            depth, min_value=self.min_value, max_value=self.max_value)
        depth_bgr = depth_bgr.astype(np.float32)
        depth_bgr -= mean_bgr
        depth_bgr = depth_bgr.transpose((2, 0, 1))

        return image_bgr, depth_nan2zero, depth_bgr, label_gt, depth_gt

    def summarizing_process(self):
        print('========================================================')
        print('Summary of segmentation and depth accuracy')

        sum_miou = 0.0
        id = 0
        for id in range(self.data_len):
            image_rgb, depth, label_gt, depth_gt = self.dataset[id]
            image_bgr, depth_nan2zero, depth_bgr, label_gt, depth_gt = \
                self._transform(image_rgb, depth, label_gt, depth_gt)
            pred_label, pred_depth = self._segment(
                image_bgr, depth_nan2zero, depth_bgr)
            pred_label = pred_label.astype(np.int32)

            # evaluate mean IU
            miou = fcn.utils.label_accuracy_score(
                [label_gt], [pred_label], n_class=2)[2]
            sum_miou += miou

        ave_miou = sum_miou / self.data_len

        print('mean IU: %lf' % ave_miou)
        print('========================================================\n')

    def process(self):
        id = 0
        while id >= 0 and id <= self.data_len:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('id: %d' % id)

            image_rgb, depth, label_gt, depth_gt = self.dataset[id]
            image_bgr, depth_nan2zero, depth_bgr, label_gt, depth_gt = \
                self._transform(image_rgb, depth, label_gt, depth_gt)
            pred_label, pred_depth = self._segment(
                image_bgr, depth_nan2zero, depth_bgr)
            pred_label = pred_label.astype(np.int32)

            # evaluate mean IU
            miou = fcn.utils.label_accuracy_score(
                [label_gt], [pred_label], n_class=2)[2]
            print('mean IU: %lf' % miou)

            # visualize depth, label, proba
            depth_rgb = self._colorize_depth(
                depth, self.min_value, self.max_value)[:, :, ::-1]
            label_gt_viz = mvtk.image.label2rgb(
                label_gt, img=image_rgb,
                label_names=self.class_names, alpha=0.7)
            depth_gt_viz = self._colorize_depth(
                depth_gt, self.min_value, self.max_value)[:, :, ::-1]
            pred_label_viz = mvtk.image.label2rgb(
                pred_label, img=image_rgb,
                label_names=self.class_names, alpha=0.7)
            pred_depth_viz = self._colorize_depth(
                pred_depth, self.min_value, self.max_value)[:, :, ::-1]

            viz = mvtk.image.tile(
                [image_rgb, pred_label_viz, label_gt_viz,
                 depth_rgb, pred_depth_viz, depth_gt_viz],
                shape=(2, 3))
            mvtk.image.resize(viz, size=300 * 300)
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            viz = cv2.resize(viz, None, None, fx=0.7, fy=0.7)
            cv2.namedWindow('viz')
            cv2.moveWindow('viz', 0, 0)
            cv2.imshow('viz', viz[:, :, ::-1])

            key = cv2.waitKey(0)
            if key == ord('q'):
                return
            elif key == ord('p') and id > 0:
                id -= 1
                continue
            elif key == ord('n') and id < self.data_len - 1:
                id += 1
                continue
            else:
                continue

    def _segment(self, image_bgr, depth, depth_bgr):
        image_bgr_data = np.array([image_bgr], dtype=np.float32)
        depth_data = np.array([depth], dtype=np.float32)
        depth_bgr_data = np.array([depth_bgr], dtype=np.float32)
        if self.gpu != -1:
            image_bgr_data = cuda.to_gpu(image_bgr_data, device=self.gpu)
            depth_data = cuda.to_gpu(depth_data, device=self.gpu)
            depth_bgr_data = cuda.to_gpu(depth_bgr_data, device=self.gpu)

        if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
            image_bgr_variable = chainer.Variable(
                image_bgr_data, volatile=True)
            depth_variable = chainer.Variable(
                depth_data, volatile=True)
            depth_bgr_variable = chainer.Variable(
                depth_bgr_data, volatile=True)
            self.model(image_bgr_variable, depth_variable, depth_bgr_variable,
                       None, None)
        else:
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    image_bgr_variable = chainer.Variable(image_bgr_data)
                    depth_variable = chainer.Variable(depth_data)
                    depth_bgr_variable = chainer.Variable(depth_bgr_data)
                    self.model(image_bgr_variable, depth_variable,
                               depth_bgr_variable, None, None)

        # Get proba_img, pred_label, pred_depth
        proba_img = F.softmax(self.model.score_label)
        proba_img = F.transpose(proba_img, (0, 2, 3, 1))
        max_proba_img = F.max(proba_img, axis=-1)
        pred_label = F.argmax(self.model.score_label, axis=1)
        pred_depth = self.model.depth_pred

        # squeeze batch axis, gpu -> cpu
        max_proba_img = cuda.to_cpu(max_proba_img.data)[0]
        pred_label = cuda.to_cpu(pred_label.data)[0]
        pred_depth = cuda.to_cpu(pred_depth.data)[0, 0]

        # uncertain because the probability is low
        pred_label[max_proba_img < self.proba_threshold] = self.bg_label

        return pred_label, pred_depth


if __name__ == '__main__':
    FCNMirrorSegmentationDepthEstimation()
