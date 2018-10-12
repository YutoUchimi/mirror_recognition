#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

from distutils.version import LooseVersion
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

from models import FCN8sMirrorObjectSegmentation  # NOQA


class FCNMirrorObjectSegmentation(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.model_dir = rospy.get_param('~model_dir')
        self.gpu = rospy.get_param('~gpu', -1)
        self.bg_label = rospy.get_param('~bg_label', 0)
        self.proba_threshold = rospy.get_param('~proba_threshold', 0.5)
        self.pub_label_mirror = self.advertise(
            '~output/label/mirror', Image, queue_size=1)
        self.pub_label_object = self.advertise(
            '~output/label/object', Image, queue_size=1)
        self.pub_proba_mirror = self.advertise(
            '~output/proba/mirror', Image, queue_size=1)
        self.pub_proba_object = self.advertise(
            '~output/proba/object', Image, queue_size=1)

        expected_file_set = set([
            'model.txt',
            'n_class.txt',
            'max_mean_iu_object.npz',
        ])
        actual_file_set = set(os.listdir(osp.expanduser(self.model_dir)))
        if not expected_file_set.issubset(actual_file_set):
            rospy.logerr(
                'File set does not match. Expected: {}, Actual: {}'.format(
                    expected_file_set, actual_file_set))

        self._load_model()

    def _load_model(self):
        with open(osp.join(self.model_dir, 'model.txt'), 'r') as f:
            model_name = f.readline().rstrip()
        with open(osp.join(self.model_dir, 'n_class.txt'), 'r') as f:
            n_class = int(f.readline().rstrip())
        self.n_class = n_class

        if model_name == 'FCN8sMirrorObjectSegmentation':
            self.model = FCN8sMirrorObjectSegmentation(n_class_object=n_class)

        model_file = osp.join(self.model_dir, 'max_mean_iu_object.npz')
        rospy.loginfo(
            '\nLoading trained model:          {0}'.format(model_file))
        S.load_npz(model_file, self.model)
        rospy.loginfo(
            'Finished loading trained model: {0}\n'.format(model_file))

        if self.gpu >= 0:
            cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
            self.model.train = False

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
        mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        bgr_img = (bgr_img - mean_bgr).transpose((2, 0, 1))

        label_mirror, label_object, proba_mirror, proba_object = \
            self._segment(bgr_img)
        label_mirror = label_mirror.astype(np.int32)
        label_object = label_object.astype(np.int32)
        proba_mirror = proba_mirror.astype(np.float32)
        proba_object = proba_object.astype(np.float32)

        label_mirror_msg = br.cv2_to_imgmsg(label_mirror, '32SC1')
        label_mirror_msg.header = img_msg.header
        self.pub_label_mirror.publish(label_mirror_msg)
        label_object_msg = br.cv2_to_imgmsg(label_object, '32SC1')
        label_object_msg.header = img_msg.header
        self.pub_label_object.publish(label_object_msg)
        proba_mirror_msg = br.cv2_to_imgmsg(proba_mirror)
        proba_mirror_msg.header = img_msg.header
        self.pub_proba_mirror.publish(proba_mirror_msg)
        proba_object_msg = br.cv2_to_imgmsg(proba_object)
        proba_object_msg.header = img_msg.header
        self.pub_proba_object.publish(proba_object_msg)

    def _segment(self, bgr):
        bgr_data = np.array([bgr], dtype=np.float32)
        if self.gpu != -1:
            bgr_data = cuda.to_gpu(bgr_data, device=self.gpu)

        if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
            bgr = chainer.Variable(bgr_data, volatile=True)
            self.model(bgr)
        else:
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    bgr = chainer.Variable(bgr_data)
                    self.model(bgr)

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
        proba_img_mirror = cuda.to_cpu(proba_img_mirror.data)[0]
        proba_img_object = cuda.to_cpu(proba_img_object.data)[0]
        max_proba_img_mirror = cuda.to_cpu(max_proba_img_mirror.data)[0]
        max_proba_img_object = cuda.to_cpu(max_proba_img_object.data)[0]
        pred_label_mirror = cuda.to_cpu(pred_label_mirror.data)[0]
        pred_label_object = cuda.to_cpu(pred_label_object.data)[0]

        # uncertain because the probability is low
        pred_label_mirror[
            max_proba_img_mirror < self.proba_threshold] = self.bg_label
        pred_label_object[
            max_proba_img_object < self.proba_threshold] = self.bg_label

        return pred_label_mirror, pred_label_object, \
            proba_img_mirror, proba_img_object


if __name__ == '__main__':
    rospy.init_node('fcn_mirror_object_segmentation')
    FCNMirrorObjectSegmentation()
    rospy.spin()
