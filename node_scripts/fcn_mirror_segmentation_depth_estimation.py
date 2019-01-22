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
from models import FCN8sMirrorSegmentationDepthEstimation  # NOQA


class FCNMirrorSegmentationDepthEstimation(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.model_dir = rospy.get_param('~model_dir')
        self.gpu = rospy.get_param('~gpu', 0)
        self.bg_label = rospy.get_param('~bg_label', 0)
        self.proba_threshold = rospy.get_param('~proba_threshold', 0.5)
        self.pub_label = self.advertise('~output/label', Image, queue_size=1)
        self.pub_proba = self.advertise('~output/proba', Image, queue_size=1)
        self.pub_depth_inpainted = self.advertise(
            '~output/depth_inpainted', Image, queue_size=1)
        self.pub_depth_pred_raw = self.advertise(
            '~output/depth_pred_raw', Image, queue_size=1)
        self.pub_depth_pred_labeled = self.advertise(
            '~output/depth_pred_labeled', Image, queue_size=1)

        self.xp = cuda.cupy if self.gpu >= 0 else np

        expected_file_set = set([
            'model.txt',
            'n_class.txt',
            'max_miou.npz',
            'max_depth_acc.npz',
        ])
        actual_file_set = set(os.listdir(osp.expanduser(self.model_dir)))
        if not expected_file_set.issubset(actual_file_set):
            rospy.logerr('File set does not match. Expected: {}, Actual: {}'.
                         format(expected_file_set, actual_file_set))

        self.model_ready = False
        self._load_model()
        self.model_ready = True

    def _load_model(self):
        with open(osp.join(self.model_dir, 'model.txt'), 'r') as f:
            model_name = f.readline().rstrip()
        with open(osp.join(self.model_dir, 'n_class.txt'), 'r') as f:
            n_class = int(f.readline().rstrip())
        self.n_class = n_class

        if model_name == 'FCN8sMirrorSegmentationDepthEstimation':
            self.model = FCN8sMirrorSegmentationDepthEstimation(
                n_class=n_class)

        # model_file = osp.join(self.model_dir, 'max_miou.npz')
        model_file = osp.join(self.model_dir, 'max_depth_acc.npz')
        rospy.loginfo('Start loading trained model:    {0}'.format(model_file))
        S.load_npz(model_file, self.model)
        rospy.loginfo('Finished loading trained model: {0}'.format(model_file))

        if self.gpu >= 0:
            self.model.to_gpu(self.gpu)

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_img = message_filters.Subscriber(
            '~input/image', Image, queue_size=1, buff_size=2**24)
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
        # Wait until model is loaded.
        if not self.model_ready:
            return

        # Model output: (0.4, 5.1) for converging at sigmoid function
        # Required model input: (0.5, 5.0)
        self.min_value = self.model.min_depth + 0.1
        self.max_value = self.model.max_depth - 0.1

        br = cv_bridge.CvBridge()
        img_bgr = br.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8').astype(np.float32)
        depth = br.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        # Preprocessing
        # BGR: (C, H, W)
        mean_bgr = np.array(
            [104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
        img_bgr_chw = (img_bgr - mean_bgr).transpose((2, 0, 1))

        # depth -> depth_limited -> depth_nan2zero: (H, W) -> (1, H, W)
        depth_limited = depth.copy()
        depth_limited[depth_limited == 0] = np.nan
        depth_limited_keep = ~np.isnan(depth_limited)
        depth_limited[depth_limited_keep] = np.maximum(
            depth_limited[depth_limited_keep], self.min_value)
        depth_limited[depth_limited_keep] = np.minimum(
            depth_limited[depth_limited_keep], self.max_value)
        depth_nan2zero = depth_limited.copy()
        depth_nan2zero[np.isnan(depth_nan2zero)] = 0.0
        depth_nan2zero = depth_nan2zero[np.newaxis, :, :]

        # depth: (3, H, W)
        depth_bgr = self._colorize_depth(
            depth,
            min_value=self.min_value, max_value=self.max_value
        ).astype(np.float32)
        depth_bgr_chw = (depth_bgr - mean_bgr).transpose((2, 0, 1))

        try:
            # Main process
            label, proba, depth_pred_raw = self._predict(
                img_bgr_chw, depth_nan2zero, depth_bgr_chw)
            label = label.astype(np.int32)
            proba = proba.astype(np.float32)

            # Inpaint depth image
            depth_inpainted = depth.copy()
            depth_inpainted[label > 0] = depth_pred_raw[label > 0]

            # Depth image only labeled as mirror
            depth_pred_labeled = depth_pred_raw.copy()
            depth_pred_labeled[label == 0] = np.nan

            # Publish
            label_msg = br.cv2_to_imgmsg(label, '32SC1')
            label_msg.header = img_msg.header
            self.pub_label.publish(label_msg)
            proba_msg = br.cv2_to_imgmsg(proba)
            proba_msg.header = img_msg.header
            self.pub_proba.publish(proba_msg)
            depth_inpainted_msg = br.cv2_to_imgmsg(depth_inpainted)
            depth_inpainted_msg.header = img_msg.header
            self.pub_depth_inpainted.publish(depth_inpainted_msg)
            depth_pred_raw_msg = br.cv2_to_imgmsg(depth_pred_raw)
            depth_pred_raw_msg.header = img_msg.header
            self.pub_depth_pred_raw.publish(depth_pred_raw_msg)
            depth_pred_labeled_msg = br.cv2_to_imgmsg(depth_pred_labeled)
            depth_pred_labeled_msg.header = img_msg.header
            self.pub_depth_pred_labeled.publish(depth_pred_labeled_msg)

        # FIXME: Somehow TypeError occurs while loading model.
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

    def _predict(self, img_bgr, depth_nan2zero, depth_bgr):
        img_bgr_batch = self.xp.array([img_bgr], dtype=self.xp.float32)
        depth_batch = self.xp.array([depth_nan2zero], dtype=self.xp.float32)
        depth_bgr_batch = self.xp.array([depth_bgr], dtype=self.xp.float32)
        if self.gpu >= 0:
            img_bgr_batch = cuda.to_gpu(img_bgr_batch, device=self.gpu)
            depth_batch = cuda.to_gpu(depth_batch, device=self.gpu)
            depth_bgr_batch = cuda.to_gpu(depth_bgr_batch, device=self.gpu)

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                img_bgr_variable = chainer.Variable(img_bgr_batch)
                depth_variable = chainer.Variable(depth_batch)
                depth_bgr_variable = chainer.Variable(depth_bgr_batch)
                # Do inference
                self.model(
                    img_bgr_variable, depth_variable, depth_bgr_variable,
                    None, None)

        # Get proba_img, pred_label, pred_depth
        proba_img = F.softmax(self.model.score_label)
        proba_img = F.transpose(proba_img, (0, 2, 3, 1))
        max_proba_img = F.max(proba_img, axis=-1)
        pred_label = F.argmax(self.model.score_label, axis=1)
        pred_depth = self.model.depth_pred

        # Squeeze batch axis, gpu -> cpu
        proba_img = cuda.to_cpu(proba_img.data)[0]
        max_proba_img = cuda.to_cpu(max_proba_img.data)[0]
        pred_label = cuda.to_cpu(pred_label.data)[0]
        pred_depth = cuda.to_cpu(pred_depth.data)[0, 0]

        # Uncertain because the probability is low
        pred_label[max_proba_img < self.proba_threshold] = self.bg_label

        return pred_label, proba_img, pred_depth


if __name__ == '__main__':
    rospy.init_node('fcn_mirror_segmentation_depth_estimation')
    FCNMirrorSegmentationDepthEstimation()
    rospy.spin()
