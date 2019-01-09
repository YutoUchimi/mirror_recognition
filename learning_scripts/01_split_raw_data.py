#!/usr/bin/env python

import argparse
import os
import os.path as osp
import shutil


def split_data(src_dir, dst_dir, ratio):
    src_dir = osp.expanduser(src_dir.rstrip('/'))
    if not osp.isdir(src_dir):
        print('[ERROR]: {} is not directory.'.format(src_dir))
        exit(1)

    dst_train_dir = osp.join(dst_dir, 'train')
    dst_test_dir = osp.join(dst_dir, 'test')

    # Make directories and check if they are empty.
    if not osp.isdir(dst_train_dir):
        print('Making {}'.format(dst_train_dir))
        os.makedirs(dst_train_dir)
    if len(os.listdir(dst_train_dir)) > 0:
        print('[ERROR]: {} is not an empty directory.'.format(dst_train_dir))
        exit(1)
    if not osp.isdir(dst_test_dir):
        print('Making {}'.format(dst_test_dir))
        os.makedirs(dst_test_dir)
    if len(os.listdir(dst_test_dir)) > 0:
        print('[ERROR]: {} is not an empty directory.'.format(dst_test_dir))
        exit(1)

    # Copy
    length = len(os.listdir(src_dir))
    for i, stamp_dir_ in enumerate(sorted(os.listdir(src_dir))):
        print('Copying {} / {}'.format(i + 1, length))
        from_stamp_dir = osp.join(src_dir, stamp_dir_)
        if i % (ratio + 1) != 0:
            shutil.copytree(
                from_stamp_dir, osp.join(dst_train_dir, stamp_dir_))
        else:
            shutil.copytree(from_stamp_dir, osp.join(dst_test_dir, stamp_dir_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_dir', type=str, required=True)
    parser.add_argument('-d', '--dst_dir', type=str, required=True)
    parser.add_argument('-r', '--ratio', type=int, required=True,
                        help='[train] : [test] = ratio : 1')
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    ratio = args.ratio

    split_data(src_dir, dst_dir, ratio)
