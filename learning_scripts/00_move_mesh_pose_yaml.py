#!/usr/bin/env python

import argparse
import glob
import os
import os.path as osp
import shutil


def move_mesh_pose_yaml(root_dir):
    # root_dir must contain timestamp data dirs and only one pose/ dir.
    # The name of timestamp dirs existing in pose/ dir may be differerent
    # from that existing in root_dir.
    root_dir = osp.expanduser(root_dir.rstrip('/'))
    if not osp.isdir(root_dir):
        print('[ERROR]: {} is not directory.'.format(root_dir))
        exit(1)

    src_dir = osp.join(root_dir, 'pose')
    dst_dir = root_dir
    if not osp.isdir(src_dir):
        print('[ERROR]: {} is not directory.'.format(src_dir))
        exit(1)

    assert len(os.listdir(src_dir)) == len(os.listdir(dst_dir)) - 1

    # Copy
    length = len(os.listdir(src_dir))
    for i, stamp_dir_ in enumerate(sorted(os.listdir(src_dir))):
        print('Moving mesh pose: {} / {}'.format(i + 1, length))
        src_stamp_dir = osp.join(src_dir, stamp_dir_)
        for f in glob.glob(osp.join(src_stamp_dir, '*')):
            shutil.copy(f, osp.join(dst_dir, sorted(os.listdir(dst_dir))[i]))

    # Remove no more needed pose/ dir
    shutil.rmtree(src_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--root_dir', type=str, required=True,
        help='Must contain timestamp data dirs and only one pose/ dir.')
    args = parser.parse_args()
    root_dir = args.root_dir

    move_mesh_pose_yaml(root_dir)
