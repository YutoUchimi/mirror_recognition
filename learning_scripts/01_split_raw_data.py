#!/usr/bin/env python

import argparse
import os
import os.path as osp
import shutil


def split_data(from_dir, ratio):
    from_dir = osp.expanduser(from_dir.rstrip('/'))
    if not osp.isdir(from_dir):
        print('[ERROR]: {} is not directory.'.format(from_dir))
        exit(1)

    to_train_dir = osp.join(osp.dirname(from_dir), 'train')
    to_test_dir = osp.join(osp.dirname(from_dir), 'test')

    # Make directories and check if they are empty.
    if not osp.isdir(to_train_dir):
        print('Making {}'.format(to_train_dir))
        os.makedirs(to_train_dir)
    if len(os.listdir(to_train_dir)) > 0:
        print('[ERROR]: {} is not an empty directory.'.format(to_train_dir))
        exit(1)
    if not osp.isdir(to_test_dir):
        print('Making {}'.format(to_test_dir))
        os.makedirs(to_test_dir)
    if len(os.listdir(to_test_dir)) > 0:
        print('[ERROR]: {} is not an empty directory.'.format(to_test_dir))
        exit(1)

    # Copy
    length = len(os.listdir(from_dir))
    for i, stamp_dir_ in enumerate(sorted(os.listdir(from_dir))):
        print('Copying {} / {}'.format(i + 1, length))
        from_stamp_dir = osp.join(from_dir, stamp_dir_)
        if i % (ratio + 1) != 0:
            shutil.copytree(from_stamp_dir, osp.join(to_train_dir, stamp_dir_))
        else:
            shutil.copytree(from_stamp_dir, osp.join(to_test_dir, stamp_dir_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--from_dir', type=str, required=True)
    parser.add_argument('-r', '--ratio', type=int, required=True,
                        help='[train] : [test] = ratio : 1')
    args = parser.parse_args()

    from_dir = args.from_dir
    ratio = args.ratio

    split_data(from_dir, ratio)
