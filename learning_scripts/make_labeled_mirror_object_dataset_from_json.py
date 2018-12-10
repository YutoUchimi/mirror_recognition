#!/usr/bin/env python

from __future__ import print_function

import argparse
import base64
import json
import os
import os.path as osp

from labelme import utils
import PIL.Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'src_dir', type=str,
        help='Input data directory. It must have [split] dirs.')
    parser.add_argument(
        'dst_dir', type=str,
        help='Output dataset directory.')
    parser.add_argument(
        'split', choices=['train', 'test'],
        help='Choose split you want to make dataset of.')
    args = parser.parse_args()

    raw_split_dir = osp.join(args.src_dir, args.split)
    out_split_dir = osp.join(args.dst_dir, args.split)
    for stamp_dir in sorted(os.listdir(raw_split_dir)):
        mirror_json_file = osp.join(raw_split_dir, stamp_dir, 'mirror.json')
        object_json_file = osp.join(raw_split_dir, stamp_dir, 'object.json')
        if not osp.exists(mirror_json_file):
            print('{} does not exist.'.format(mirror_json_file))
            exit(1)
        if not osp.exists(object_json_file):
            print('{} does not exist.'.format(object_json_file))
            exit(1)

        out_dir = osp.join(out_split_dir, stamp_dir)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        save_image_from_json(mirror_json_file, out_dir)
        save_label_from_json(mirror_json_file, out_dir, target='mirror')
        save_label_from_json(object_json_file, out_dir, target='object')
        print('Saved to: %s' % out_dir)


def save_image_from_json(json_file, out_dir):
    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = osp.join(osp.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'image.png'))


def save_label_from_json(json_file, out_dir, target):
    assert target in ['mirror', 'object']

    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = osp.join(osp.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': 0}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = utils.draw_label(lbl, img, label_names)

    utils.lblsave(osp.join(out_dir, target + '_label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(
        osp.join(out_dir, target + '_label_viz.png'))

    with open(osp.join(out_dir, target + '_label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')


if __name__ == '__main__':
    main()
