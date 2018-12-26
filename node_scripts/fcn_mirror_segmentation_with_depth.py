#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import os
import os.path as osp
import sys
import traceback

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.serializers as S
import cv2
import numpy as np

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import message_filters
import rospy
from sensor_msgs.msg import Image

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from models import FCN8sMirrorSegmentationWithDepth  # NOQA


class FCNMirrorSegmentationWithDepth(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.model_dir = rospy.get_param('~model_dir')
        self.gpu = rospy.get_param('~gpu', 0)
        self.bg_label = rospy.get_param('~bg_label', 0)
        self.proba_threshold = rospy.get_param('~proba_threshold', 0.5)
        self.pub_label = self.advertise('~output/label', Image, queue_size=1)
        self.pub_proba = self.advertise('~output/proba', Image, queue_size=1)

        self.xp = cuda.cupy if self.gpu >= 0 else np

        expected_file_set = set([
            'model.txt',
            'n_class.txt',
            'max_miou.npz',
        ])
        actual_file_set = set(os.listdir(osp.expanduser(self.model_dir)))
        if not expected_file_set.issubset(actual_file_set):
            rospy.logerr('File set does not match. Expected: {}, Actual: {}'.
                         format(expected_file_set, actual_file_set))

        self._load_model()

    def _load_model(self):
        with open(osp.join(self.model_dir, 'model.txt'), 'r') as f:
            model_name = f.readline().rstrip()
        with open(osp.join(self.model_dir, 'n_class.txt'), 'r') as f:
            n_class = int(f.readline().rstrip())
        self.n_class = n_class

        if model_name == 'FCN8sMirrorSegmentationWithDepth':
            self.model = FCN8sMirrorSegmentationWithDepth(n_class=n_class)

        model_file = osp.join(self.model_dir, 'max_miou.npz')
        rospy.loginfo('Loading trained model:          {0}'.format(model_file))
        S.load_npz(model_file, self.model)
        rospy.loginfo('Finished loading trained model: {0}'.format(model_file))

        if self.gpu >= 0:
            self.model.to_gpu(self.gpu)

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_img = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2**24)
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2**24)
        self.subs = [sub_img, sub_depth]

        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self._cb)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _cb(self, img_msg, depth_msg):
        br = cv_bridge.CvBridge()
        mean_bgr = np.array(
            [104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

        img_bgr = br.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8').astype(np.float32)
        img_bgr_chw = (img_bgr - mean_bgr).transpose((2, 0, 1))

        depth = br.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        depth_bgr = self._colorize_depth(depth).astype(np.float32)
        depth_bgr_chw = (depth_bgr - mean_bgr).transpose((2, 0, 1))

        try:
            label, proba = self._segment(img_bgr_chw, depth_bgr_chw)
            label = label.astype(np.int32)
            proba = proba.astype(np.float32)

            label_msg = br.cv2_to_imgmsg(label, '32SC1')
            label_msg.header = img_msg.header
            self.pub_label.publish(label_msg)
            proba_msg = br.cv2_to_imgmsg(proba)
            proba_msg.header = img_msg.header
            self.pub_proba.publish(proba_msg)

        except TypeError:
            rospy.logdebug(traceback.format_exc())

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

    def _segment(self, img_bgr, depth_bgr):
        img_bgr_batch = self.xp.array([img_bgr], dtype=self.xp.float32)
        depth_bgr_batch = self.xp.array([depth_bgr], dtype=self.xp.float32)
        if self.gpu >= 0:
            img_bgr_batch = cuda.to_gpu(img_bgr_batch, device=self.gpu)
            depth_bgr_batch = cuda.to_gpu(depth_bgr_batch, device=self.gpu)

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                img_bgr_variable = chainer.Variable(img_bgr_batch)
                depth_bgr_variable = chainer.Variable(depth_bgr_batch)
                self.model(img_bgr_variable, depth_bgr_variable)

        # Get proba_img, pred_label
        proba_img = F.softmax(self.model.score)
        proba_img = F.transpose(proba_img, (0, 2, 3, 1))
        max_proba_img = F.max(proba_img, axis=-1)
        pred_label = F.argmax(self.model.score, axis=1)

        # squeeze batch axis, gpu -> cpu
        proba_img = cuda.to_cpu(proba_img.data)[0]
        max_proba_img = cuda.to_cpu(max_proba_img.data)[0]
        pred_label = cuda.to_cpu(pred_label.data)[0]

        # uncertain because the probability is low
        pred_label[max_proba_img < self.proba_threshold] = self.bg_label

        return pred_label, proba_img


if __name__ == '__main__':
    rospy.init_node('fcn_mirror_segmentation_with_depth')
    FCNMirrorSegmentationWithDepth()
    rospy.spin()
