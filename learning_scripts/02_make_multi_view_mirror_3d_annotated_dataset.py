#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import os.path as osp
import shutil
import sys

import labelme.utils
import numpy as np
import skimage.io
import yaml

sys.path.append(osp.dirname(osp.abspath(__file__)))
import utils  # NOQA


def save_image(image_file, out_dir):
    image = skimage.io.imread(image_file)
    skimage.io.imsave(osp.join(out_dir, 'image.png'), image)


def copy_raw_depth(depth_file, out_dir):
    shutil.copy(depth_file, osp.join(out_dir, 'depth.npz'))


def copy_scene_id(scene_id_file, out_dir):
    shutil.copy(scene_id_file, osp.join(out_dir, 'scene_id.txt'))


def copy_base_offset_x(base_offset_x_file, out_dir):
    shutil.copy(base_offset_x_file, osp.join(out_dir, 'base_offset_x.txt'))


def copy_base_offset_y(base_offset_y_file, out_dir):
    shutil.copy(base_offset_y_file, osp.join(out_dir, 'base_offset_y.txt'))


def copy_head_offset_p(head_offset_p_file, out_dir):
    shutil.copy(head_offset_p_file, osp.join(out_dir, 'head_offset_p.txt'))


def copy_head_offset_y(head_offset_y_file, out_dir):
    shutil.copy(head_offset_y_file, osp.join(out_dir, 'head_offset_y.txt'))


def copy_tf_base_to_cam(tf_base_to_cam_file, out_dir):
    shutil.copy(tf_base_to_cam_file,
                osp.join(out_dir, 'tf_base_to_camera.yaml'))


def copy_tf_map_to_cam(tf_map_to_cam_file, out_dir):
    shutil.copy(tf_map_to_cam_file,
                osp.join(out_dir, 'tf_map_to_camera.yaml'))


def get_mesh(mesh_id):
    shapenet_root = osp.expanduser('~/data/ShapeNetCore.v2.scaled')
    off_files = osp.join(
        shapenet_root, '%s/models/model_normalized_scaled.off' % mesh_id)
    vertices, faces = utils.load_off(off_files)
    return vertices, faces


def save_generated_depth_gt_label_gt(
        depth_file, cam_info_file, tf_camera_to_obj_file, mesh_id_file,
        out_dir):
    # Load files
    depth = np.load(depth_file)['arr_0']
    cam_info = yaml.load(open(cam_info_file))
    K = np.asarray(cam_info['K']).reshape(3, 3)
    tf_camera_to_obj = utils.transform_to_matrix(tf_camera_to_obj_file)
    mesh_id = yaml.load(open(mesh_id_file))['mesh_id']

    # Generate depth and mask of mirror
    height, width = depth.shape[:2]
    vertices, faces = get_mesh(mesh_id)
    depth_mirror, mask_mirror = utils.generate_depth(
        vertices, faces, height, width, tf_camera_to_obj, K)
    assert depth_mirror.shape == depth.shape
    assert depth_mirror.dtype == np.float32 or \
        depth_mirror.dtype == np.float64
    assert mask_mirror.shape == depth.shape
    assert mask_mirror.dtype == np.bool

    # Get min(raw_depth, gen_depth) in mirror region
    depth_nan2inf = depth.copy()
    depth_nan2inf[np.isnan(depth_nan2inf)] = np.inf
    depth_mirror_nan2inf = depth_mirror.copy()
    depth_mirror_nan2inf[np.isnan(depth_mirror_nan2inf)] = np.inf
    depth_gt = depth.copy()
    depth_gt[mask_mirror] = np.minimum(
        depth_nan2inf[mask_mirror],
        depth_mirror_nan2inf[mask_mirror],
    )
    depth_gt = depth_gt.astype(np.float32)

    # Update mask_mirror according to depth_gt, and convert it to label image
    mask_gt = mask_mirror.copy()
    mask_gt[depth_nan2inf < depth_gt] = False
    label_gt = mask_gt.astype(np.int32)

    # Save generated depth
    depth_gt_npz = osp.join(out_dir, 'depth_gt.npz')
    np.savez_compressed(depth_gt_npz, depth_gt)

    # Save generated label
    labelme.utils.lblsave(osp.join(out_dir, 'label.png'), label_gt)


def main(src_dir, dst_dir):
    length = len(os.listdir(src_dir))
    for i, stamp_dir in enumerate(sorted(os.listdir(src_dir))):
        # Check if required files exist.
        image_file = osp.join(src_dir, stamp_dir, 'image.jpg')
        depth_file = osp.join(src_dir, stamp_dir, 'depth.npz')
        cam_info_file = osp.join(src_dir, stamp_dir, 'camera_info.yaml')
        tf_camera_to_obj_file = osp.join(
            src_dir, stamp_dir, 'tf_camera_to_obj.yaml')
        mesh_id_file = osp.join(src_dir, stamp_dir, 'mesh_id.yaml')
        scene_id_file = osp.join(src_dir, stamp_dir, 'scene_id.txt')
        base_offset_x_file = osp.join(src_dir, stamp_dir, 'base_offset_x.txt')
        base_offset_y_file = osp.join(src_dir, stamp_dir, 'base_offset_y.txt')
        head_offset_p_file = osp.join(src_dir, stamp_dir, 'head_offset_p.txt')
        head_offset_y_file = osp.join(src_dir, stamp_dir, 'head_offset_y.txt')
        tf_base_to_cam_file = osp.join(
            src_dir, stamp_dir, 'tf_base_to_camera.yaml')
        tf_map_to_cam_file = osp.join(
            src_dir, stamp_dir, 'tf_map_to_camera.yaml')
        required_files = [
            image_file,
            depth_file,
            cam_info_file,
            tf_camera_to_obj_file,
            mesh_id_file,
            scene_id_file,
            base_offset_x_file,
            base_offset_y_file,
            head_offset_p_file,
            head_offset_y_file,
            tf_base_to_cam_file,
            tf_map_to_cam_file
        ]
        for f in required_files:
            if not osp.exists(image_file):
                print('{} does not exist.'.format(f))
                exit(1)

        # Make output directory
        out_dir = osp.join(dst_dir, stamp_dir)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        # Main process
        save_image(image_file, out_dir)
        copy_raw_depth(depth_file, out_dir)
        copy_scene_id(scene_id_file, out_dir)
        copy_base_offset_x(base_offset_x_file, out_dir)
        copy_base_offset_y(base_offset_y_file, out_dir)
        copy_head_offset_p(head_offset_p_file, out_dir)
        copy_head_offset_y(head_offset_y_file, out_dir)
        copy_tf_base_to_cam(tf_base_to_cam_file, out_dir)
        copy_tf_map_to_cam(tf_map_to_cam_file, out_dir)
        save_generated_depth_gt_label_gt(
            depth_file, cam_info_file, tf_camera_to_obj_file, mesh_id_file,
            out_dir)
        print('[%d / %d] Saved to: %s' % (i + 1, length, out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--src_dir', type=str, required=True,
        help='Input annotating data directory. It must have timestamp dirs.')
    parser.add_argument(
        '-d', '--dst_dir', type=str, required=True,
        help='Output dataset directory.')

    args = parser.parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    main(src_dir, dst_dir)
