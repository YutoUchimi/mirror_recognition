#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import argparse
import base64
import json
import os
import os.path as osp
import shutil
import sys

import labelme.utils
import numpy as np
import PIL.Image
import yaml

sys.path.append(osp.dirname(osp.abspath(__file__)))
import utils  # NOQA


def save_image_from_json(json_file, out_dir):
    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = osp.join(osp.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = labelme.utils.img_b64_to_arr(imageData)
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'image.png'))


def save_label_from_json(json_file, out_dir):
    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = osp.join(osp.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = labelme.utils.img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = labelme.utils.shapes_to_label(
        img.shape, data['shapes'], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = labelme.utils.draw_label(lbl, img, label_names)

    labelme.utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(
        osp.join(out_dir, 'label_viz.png'))

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')


def copy_raw_depth(depth_file, out_dir):
    shutil.copy(depth_file, osp.join(out_dir, 'depth.npz'))


def get_mesh(mesh_id):
    shapenet_root = osp.expanduser('~/data/ShapeNetCore.v2.scaled')
    off_files = osp.join(
        shapenet_root, '%s/models/model_normalized_scaled.off' % mesh_id)
    vertices, faces = utils.load_off(off_files)
    return vertices, faces


def save_generated_depth_gt(depth_file, cam_info_file, tf_camera_to_obj_file,
                            mesh_id_file, out_dir):
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
    depth_gt = depth.copy()
    depth_mirror_nan2inf = depth_mirror.copy()
    depth_mirror_nan2inf[np.isnan(depth_mirror_nan2inf)] = np.inf
    depth_gt[mask_mirror] = np.minimum(
        depth_mirror_nan2inf[mask_mirror],
        depth[mask_mirror],
    )
    depth_gt = depth_gt.astype(np.float32)

    # Save generated depth
    depth_gt_npz = osp.join(out_dir, 'depth_gt.npz')
    np.savez_compressed(depth_gt_npz, depth_gt)


def main(src_dir, dst_dir, split):
    raw_split_dir = osp.join(src_dir, split)
    out_split_dir = osp.join(dst_dir, split)
    length = len(os.listdir(raw_split_dir))
    for i, stamp_dir in enumerate(sorted(os.listdir(raw_split_dir))):
        # Check if required files exist.
        json_file = osp.join(raw_split_dir, stamp_dir, 'image.json')
        depth_file = osp.join(raw_split_dir, stamp_dir, 'depth.npz')
        cam_info_file = osp.join(raw_split_dir, stamp_dir, 'camera_info.yaml')
        tf_camera_to_obj_file = osp.join(
            raw_split_dir, stamp_dir, 'tf_camera_to_obj.yaml')
        mesh_id_file = osp.join(raw_split_dir, stamp_dir, 'mesh_id.yaml')
        required_files = [
            json_file,
            depth_file,
            cam_info_file,
            tf_camera_to_obj_file,
            mesh_id_file
        ]
        for f in required_files:
            if not osp.exists(json_file):
                print('{} does not exist.'.format(f))
                exit(1)

        # Make output directory
        out_dir = osp.join(out_split_dir, stamp_dir)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        # Main process
        save_image_from_json(json_file, out_dir)
        save_label_from_json(json_file, out_dir)
        copy_raw_depth(depth_file, out_dir)
        save_generated_depth_gt(
            depth_file, cam_info_file, tf_camera_to_obj_file, mesh_id_file,
            out_dir)
        print('[%d / %d] Saved to: %s' % (i + 1, length, out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--src_dir', type=str, required=True,
        help='Input annotating data directory. It must have [split] dirs.')
    parser.add_argument(
        '-d', '--dst_dir', type=str, required=True,
        help='Output dataset directory.')

    args = parser.parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    for split in ['train', 'test']:
        print('Starting split: {}'.format(split))
        main(src_dir, dst_dir, split)
