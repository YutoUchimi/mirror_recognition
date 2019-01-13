#!/usr/bin/env python

from __future__ import print_function

import json
import os
import os.path as osp
import warnings

import pandas as pd
import tabulate


def main():
    here = osp.dirname(osp.abspath(__file__))
    logs_dir = osp.join(osp.dirname(here), 'logs')

    headers = [
        'log_dir',
        'epoch',
        'iter',
        'train/miou',
        'train/depth_acc<0.03',
        'train/depth_acc<0.10',
        'train/depth_acc<0.30',
        'epoch',
        'iter',
        'val/miou',
        'val/depth_acc<0.03',
        'val/depth_acc<0.10',
        'val/depth_acc<0.30',
    ]
    rows = []

    for log in os.listdir(logs_dir):
        log_dir = osp.join(logs_dir, log)
        if not osp.isdir(log_dir):
            continue

        log_file = osp.join(log_dir, 'log.json')
        try:
            df = pd.DataFrame(json.load(open(log_file)))
        except Exception as e:
            continue

        # df = pd.read_json(log_file)
        df = df.set_index(['epoch', 'iteration'])

        # train
        columns = [
            'main/miou',
            'main/depth_acc<0.03',
            'main/depth_acc<0.10',
            'main/depth_acc<0.30',
        ]
        try:
            df_train = df[columns]
        except Exception as e:
            print(e)
            continue
        df_train = df_train.dropna()

        # validation
        columns = [
            'validation/main/miou',
            'validation/main/depth_acc<0.03',
            'validation/main/depth_acc<0.10',
            'validation/main/depth_acc<0.30',
        ]
        try:
            df_val = df[columns]
        except Exception as e:
            print(e)
            continue
        df_val = df_val.dropna()

        index_best_train = df_train['main/depth_acc<0.03'].idxmax()
        index_best_val = df_val['validation/main/depth_acc<0.03'].idxmax()
        sort_column = 11

        warnings.filterwarnings('ignore', category=UnicodeWarning)
        row_best_train = df_train.ix[index_best_train]
        row_best_val = df_val.ix[index_best_val]
        rows.append([
            log,
            index_best_train[0],
            index_best_train[1],
            row_best_train['main/miou'],
            row_best_train['main/depth_acc<0.03'],
            row_best_train['main/depth_acc<0.10'],
            row_best_train['main/depth_acc<0.30'],
            index_best_val[0],
            index_best_val[1],
            row_best_val['validation/main/miou'],
            row_best_val['validation/main/depth_acc<0.03'],
            row_best_val['validation/main/depth_acc<0.10'],
            row_best_val['validation/main/depth_acc<0.30'],
        ])
    rows.sort(key=lambda x: x[sort_column], reverse=True)
    print(tabulate.tabulate(rows, headers=headers))


if __name__ == '__main__':
    main()
