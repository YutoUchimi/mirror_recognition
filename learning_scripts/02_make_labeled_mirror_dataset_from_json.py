#!/usr/bin/env python

from __future__ import print_function

import argparse
import base64
import json
import os
import os.path as osp

from labelme import utils
import PIL.Image


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


def save_label_from_json(json_file, out_dir):
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

    utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(
        osp.join(out_dir, 'label_viz.png'))

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')


def main(src_dir, dst_dir, split):
    raw_split_dir = osp.join(src_dir, split)
    out_split_dir = osp.join(dst_dir, split)
    for stamp_dir in sorted(os.listdir(raw_split_dir)):
        json_file = osp.join(raw_split_dir, stamp_dir, 'image.json')
        if not osp.exists(json_file):
            print('{} does not exist.'.format(json_file))
            exit(1)

        out_dir = osp.join(out_split_dir, stamp_dir)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        save_image_from_json(json_file, out_dir)
        save_label_from_json(json_file, out_dir)
        print('Saved to: %s' % out_dir)


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
        main(src_dir, dst_dir, split)
