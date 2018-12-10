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

from datasets import LabeledMirrorObjectDataset  # NOQA
from models import FCN8sMirrorObjectSegmentation  # NOQA


class FCNMirrorObjectSegmentation(object):

    mirror_class_names = np.array([
        '_background_',
        'mirror',
    ], dtype=np.str)
    object_class_names = np.array([
        '_background_',
        '_tote_',
        'band_aid_tape',
        'bath_sponge',
        'burts_bees_baby_wipes',
        'colgate_toothbrush_4pk',
        'crayons',
        'duct_tape',
        'expo_eraser',
        'hanes_socks',
        'laugh_out_loud_jokes',
        'tissue_box',
    ], dtype=np.str)
    mirror_class_names.setflags(write=0)
    object_class_names.setflags(write=0)

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
                osp.expanduser(
                    '~mvtk/experiments/mirror_pose_estimation/logs'),
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
        model_file = osp.join(self.data_dir, 'max_mean_iu_object.npz')
        with open(osp.join(self.data_dir, 'model.txt'), 'r') as f:
            model_name = f.readline().rstrip()
        with open(osp.join(self.data_dir, 'n_class.txt'), 'r') as f:
            n_class = int(f.readline().rstrip())
        self.n_class = n_class

        if model_name == 'FCN8sMirrorObjectSegmentation':
            self.model = FCN8sMirrorObjectSegmentation(n_class_object=n_class)

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
        if dataset_name == 'LabeledMirrorObjectDataset':
            self.dataset = LabeledMirrorObjectDataset(
                split=self.split, aug=False)
        self.data_len = len(self.dataset)
        print('dataset class name: %s' % self.dataset.__class__.__name__)
        print('split: %s' % self.split)
        print('dataset length: %d' % self.data_len)
        print('n_class: %d' % self.n_class)

    def _transform(self, rgb_img, gt_mirror, gt_object):
        # RGB -> BGR
        bgr_img = rgb_img[:, :, ::-1]
        bgr_img = bgr_img.astype(np.float32)
        mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        bgr_img -= mean_bgr
        # H, W, C -> C, H, W
        bgr_img = bgr_img.transpose((2, 0, 1))
        return bgr_img, gt_mirror, gt_object

    def summarizing_process(self):
        print('========================================================')
        print('Summary of segmentation and depth accuracy')

        sum_mean_iu_mirror = 0.0
        sum_mean_iu_object = 0.0
        id = 0
        for id in range(self.data_len):
            rgb_img, gt_mirror, gt_object = self.dataset[id]
            bgr_img, gt_mirror, gt_object = self._transform(
                rgb_img, gt_mirror, gt_object)
            pred_label_mirror, pred_label_object = self._segment(bgr_img)
            pred_label_mirror = pred_label_mirror.astype(np.int32)
            pred_label_object = pred_label_object.astype(np.int32)

            # evaluate mean IU
            mean_iu_mirror = fcn.utils.label_accuracy_score(
                [gt_mirror], [pred_label_mirror], n_class=2)[2]
            mean_iu_object = fcn.utils.label_accuracy_score(
                [gt_object], [pred_label_object], n_class=self.n_class)[2]
            sum_mean_iu_mirror += mean_iu_mirror
            sum_mean_iu_object += mean_iu_object

        ave_mean_iu_mirror = sum_mean_iu_mirror / self.data_len
        ave_mean_iu_object = sum_mean_iu_object / self.data_len

        print('mean IU (mirror): %lf' % ave_mean_iu_mirror)
        print('mean IU (object): %lf' % ave_mean_iu_object)
        print('========================================================\n')

    def process(self):
        id = 0
        while id >= 0 and id <= self.data_len:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('id: %d' % id)

            rgb_img, gt_mirror, gt_object = self.dataset[id]
            bgr_img, gt_mirror, gt_object = self._transform(
                rgb_img, gt_mirror, gt_object)
            pred_label_mirror, pred_label_object = self._segment(bgr_img)
            pred_label_mirror = pred_label_mirror.astype(np.int32)
            pred_label_object = pred_label_object.astype(np.int32)

            # evaluate mean IU
            mean_iu_mirror = fcn.utils.label_accuracy_score(
                [gt_mirror], [pred_label_mirror], n_class=2)[2]
            mean_iu_object = fcn.utils.label_accuracy_score(
                [gt_object], [pred_label_object], n_class=self.n_class)[2]
            print('mean IU (mirror): %lf' % mean_iu_mirror)
            print('mean IU (object): %lf' % mean_iu_object)

            # visualize label, proba
            gt_mirror_viz = mvtk.image.label2rgb(
                gt_mirror, img=rgb_img,
                label_names=self.mirror_class_names, alpha=0.7)
            gt_object_viz = mvtk.image.label2rgb(
                gt_object, img=rgb_img,
                label_names=self.object_class_names, alpha=0.7)
            pred_label_mirror_viz = mvtk.image.label2rgb(
                pred_label_mirror, img=rgb_img,
                label_names=self.mirror_class_names, alpha=0.7)
            pred_label_object_viz = mvtk.image.label2rgb(
                pred_label_object, img=rgb_img,
                label_names=self.object_class_names, alpha=0.7)

            random_image = np.random.randint(
                0, 256, size=rgb_img.shape, dtype=np.uint8)
            viz = mvtk.image.tile(
                [rgb_img, pred_label_mirror_viz, pred_label_object_viz,
                 random_image, gt_mirror_viz, gt_object_viz],
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

    def _segment(self, bgr):
        bgr_data = np.array([bgr], dtype=np.float32)
        if self.gpu != -1:
            bgr_data = cuda.to_gpu(bgr_data, device=self.gpu)

        if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
            bgr = chainer.Variable(bgr_data, volatile=True)
            self.model(bgr, None, None)
        else:
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    bgr = chainer.Variable(bgr_data)
                    self.model(bgr, None, None)

        # Get proba_img, pred_label
        proba_img_mirror = F.softmax(self.model.score_mirror)
        proba_img_mirror = F.transpose(proba_img_mirror, (0, 2, 3, 1))
        max_proba_img_mirror = F.max(proba_img_mirror, axis=-1)

        proba_img_object = F.softmax(self.model.score_object)
        proba_img_object = F.transpose(proba_img_object, (0, 2, 3, 1))
        max_proba_img_object = F.max(proba_img_object, axis=-1)

        pred_label_mirror = F.argmax(self.model.score_mirror, axis=1)
        pred_label_object = F.argmax(self.model.score_object, axis=1)

        # squeeze batch axis, gpu -> cpu
        max_proba_img_mirror = cuda.to_cpu(max_proba_img_mirror.data)[0]
        max_proba_img_object = cuda.to_cpu(max_proba_img_object.data)[0]
        pred_label_mirror = cuda.to_cpu(pred_label_mirror.data)[0]
        pred_label_object = cuda.to_cpu(pred_label_object.data)[0]

        # uncertain because the probability is low
        pred_label_mirror[
            max_proba_img_mirror < self.proba_threshold] = self.bg_label
        pred_label_object[
            max_proba_img_object < self.proba_threshold] = self.bg_label

        return pred_label_mirror, pred_label_object


if __name__ == '__main__':
    FCNMirrorObjectSegmentation()
