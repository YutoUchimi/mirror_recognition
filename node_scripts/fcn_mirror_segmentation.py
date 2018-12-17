#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import os
import os.path as osp
import sys

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.serializers as S
import numpy as np

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import rospy
from sensor_msgs.msg import Image

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from models import FCN8sMirrorSegmentation  # NOQA


class FCNMirrorSegmentation(ConnectionBasedTransport):

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

        if model_name == 'FCN8sMirrorSegmentation':
            self.model = FCN8sMirrorSegmentation(n_class=n_class)

        model_file = osp.join(self.model_dir, 'max_miou.npz')
        rospy.loginfo('Loading trained model:          {0}'.format(model_file))
        S.load_npz(model_file, self.model)
        rospy.loginfo('Finished loading trained model: {0}'.format(model_file))

        if self.gpu >= 0:
            self.model.to_gpu(self.gpu)

    def subscribe(self):
        sub_img = rospy.Subscriber(
            '~input', Image, self._cb, queue_size=1, buff_size=2**24)
        self.subs = [sub_img]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _cb(self, img_msg):
        br = cv_bridge.CvBridge()
        bgr_img = br.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8').astype(np.float32)
        mean_bgr = np.array(
            [104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
        bgr_chw = (bgr_img - mean_bgr).transpose((2, 0, 1))

        label, proba = self._segment(bgr_chw)
        label = label.astype(np.int32)
        proba = proba.astype(np.float32)

        label_msg = br.cv2_to_imgmsg(label, '32SC1')
        label_msg.header = img_msg.header
        self.pub_label.publish(label_msg)
        proba_msg = br.cv2_to_imgmsg(proba)
        proba_msg.header = img_msg.header
        self.pub_proba.publish(proba_msg)

    def _segment(self, bgr):
        bgr_batch = self.xp.array([bgr], dtype=self.xp.float32)
        if self.gpu >= 0:
            bgr_batch = cuda.to_gpu(bgr_batch, device=self.gpu)

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                bgr_variable = chainer.Variable(bgr_batch)
                self.model(bgr_variable)

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
    rospy.init_node('fcn_mirror_segmentation')
    FCNMirrorSegmentation()
    rospy.spin()
