#!/usr/bin/env python

from __future__ import print_function

import argparse
import json
import os
import os.path as osp
import warnings

import pandas as pd
import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_key', default='mirror',
                        choices=['mirror', 'object'])
    args = parser.parse_args()
    max_key = args.max_key

    here = osp.dirname(osp.abspath(__file__))
    logs_dir = osp.join(osp.dirname(here), 'logs')

    headers = [
        'log_dir',
        'epoch',
        'iter',
        'train/mean_iu_mirror',
        'train/mean_iu_object',
        'epoch',
        'iter',
        'val/mean_iu_mirror',
        'val/mean_iu_object',
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
        columns = ['main/mean_iu_mirror', 'main/mean_iu_object']
        try:
            df_train = df[columns]
        except Exception as e:
            print(e)
            continue
        df_train = df_train.dropna()

        # validation
        columns = ['validation/main/mean_iu_mirror',
                   'validation/main/mean_iu_object']
        try:
            df_val = df[columns]
        except Exception as e:
            print(e)
            continue
        df_val = df_val.dropna()

        if max_key == 'mirror':
            index_best_train = df_train['main/mean_iu_mirror'].idxmax()
            index_best_val = df_val['validation/main/mean_iu_mirror'].idxmax()
            sort_column = 8
        elif max_key == 'object':
            index_best_train = df_train['main/mean_iu_object'].idxmax()
            index_best_val = df_val['validation/main/mean_iu_object'].idxmax()
            sort_column = 9
        else:
            raise KeyError('Unsupported max_key')

        warnings.filterwarnings('ignore', category=UnicodeWarning)
        row_best_train = df_train.ix[index_best_train]
        row_best_val = df_val.ix[index_best_val]
        rows.append([
            log,
            index_best_train[0],
            index_best_train[1],
            row_best_train['main/mean_iu_mirror'],
            row_best_train['main/mean_iu_object'],
            index_best_val[0],
            index_best_val[1],
            row_best_val['validation/main/mean_iu_mirror'],
            row_best_val['validation/main/mean_iu_object'],
        ])
    rows.sort(key=lambda x: x[sort_column], reverse=True)
    print(tabulate.tabulate(rows, headers=headers))


if __name__ == '__main__':
    main()
