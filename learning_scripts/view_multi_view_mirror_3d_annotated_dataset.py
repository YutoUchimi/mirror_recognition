#!/usr/bin/env python

import argparse
import os.path as osp
import sys

import mvtk

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from datasets import MultiViewMirror3DAnnotatedDataset  # NOQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', required=True,
                        choices=['train', 'test'])
    parser.add_argument('-a', '--aug', type=int, default=0,
                        help='Enable data augmentation if this is set to 1')
    parser.add_argument('-n', '--num_view', type=int, default=1,
                        help='Number of getting example at the same time')
    args = parser.parse_args()

    dataset = MultiViewMirror3DAnnotatedDataset(
        args.split, aug=args.aug, num_view=args.num_view)
    mvtk.datasets.view_dataset(
        dataset, MultiViewMirror3DAnnotatedDataset.visualize)


if __name__ == '__main__':
    main()
