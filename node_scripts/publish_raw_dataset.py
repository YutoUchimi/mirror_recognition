#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys

import numpy as np
import skimage.io
import yaml

import cv_bridge
import dynamic_reconfigure.server
import genpy.message
from geometry_msgs.msg import TransformStamped
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import tf

from mirror_recognition.cfg import PublishRawDatasetConfig


class RawDataset(object):

    def __init__(self):
        if len(sys.argv) < 2:
            print("usage: publish_raw_dataset.py DATA_DIR")

        # We assume /path/to/<self.root_dir>/<stamp_dir_>
        self.root_dir = osp.expanduser(sys.argv[1])
        self.stamp_dirs = []
        for stamp_dir_ in sorted(os.listdir(self.root_dir)):
            stamp_dir = osp.join(self.root_dir, stamp_dir_)
            self.stamp_dirs.append(stamp_dir)

    def __len__(self):
        return len(self.stamp_dirs)

    def get_frame(self, i):
        assert 0 <= i < len(self.stamp_dirs)

        stamp_dir = self.stamp_dirs[i]
        image = skimage.io.imread(osp.join(stamp_dir, 'image.jpg'))
        depth = np.load(osp.join(stamp_dir, 'depth.npz'))['arr_0']
        camera_info = yaml.load(open(osp.join(stamp_dir, 'camera_info.yaml')))
        tf_map_to_camera = yaml.load(open(osp.join(
            stamp_dir, 'tf_map_to_camera.yaml')))
        tf_base_to_camera = yaml.load(open(osp.join(
            stamp_dir, 'tf_base_to_camera.yaml')))

        return image, depth, camera_info, tf_map_to_camera, tf_base_to_camera


class PublishRawDataset(object):

    def __init__(self):
        self._dataset = RawDataset()

        self._config_srv = dynamic_reconfigure.server.Server(
            PublishRawDatasetConfig, self._config_cb)

        self.pub_rgb = rospy.Publisher(
            '~output/rgb/image_rect_color', Image, queue_size=1)
        self.pub_rgb_cam_info = rospy.Publisher(
            '~output/rgb/camera_info', CameraInfo, queue_size=1)
        self.pub_depth = rospy.Publisher(
            '~output/depth_registered/image_rect', Image, queue_size=1)
        self.pub_depth_cam_info = rospy.Publisher(
            '~output/depth_registered/camera_info', CameraInfo, queue_size=1)

        self.tfb = tf.broadcaster.TransformBroadcaster()
        rate = rospy.get_param('~rate', 30.0)
        self._timer = rospy.Timer(rospy.Duration(1. / rate), self._timer_cb)

    def _config_cb(self, config, level):
        self._stamp = config.scene_idx
        return config

    def _timer_cb(self, event):
        img, depth, cam_info, tf_map_to_camera, _ = self._dataset.get_frame(
            self._stamp)

        # Use current timestamp for each message
        # Camera info
        cam_info_msg = CameraInfo()
        genpy.message.fill_message_args(cam_info_msg, cam_info)
        cam_info_msg.header.stamp = event.current_real

        # TF map -> camera
        tf_msg = TransformStamped()
        genpy.message.fill_message_args(tf_msg, tf_map_to_camera)
        tf_msg.header.stamp = event.current_real

        bridge = cv_bridge.CvBridge()

        # RGB image
        imgmsg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
        imgmsg.header.frame_id = cam_info_msg.header.frame_id
        imgmsg.header.stamp = event.current_real

        # Depth image
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32)
            depth *= 0.001
        depth_msg = bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header.frame_id = cam_info_msg.header.frame_id
        depth_msg.header.stamp = event.current_real

        self.tfb.sendTransformMessage(tf_msg)
        self.pub_rgb.publish(imgmsg)
        self.pub_rgb_cam_info.publish(cam_info_msg)
        self.pub_depth.publish(depth_msg)
        self.pub_depth_cam_info.publish(cam_info_msg)


if __name__ == '__main__':
    rospy.init_node('publish_raw_dataset')
    app = PublishRawDataset()
    rospy.spin()
