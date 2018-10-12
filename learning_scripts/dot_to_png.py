#!/usr/bin/env python

from __future__ import print_function

import os
import os.path as osp
import subprocess


def dot_to_png():
    here = osp.dirname(osp.abspath(__file__))
    logs_dir = osp.join(osp.dirname(here), 'logs')

    for log in sorted(os.listdir(logs_dir)):
        log_dir = osp.join(logs_dir, log)
        dot_file = osp.join(log_dir, 'graph.dot')
        png_file = osp.join(log_dir, 'graph.png')

        if (not osp.isdir(log_dir)) or \
           (not osp.exists(dot_file)) or \
           osp.exists(png_file):
            continue

        print('Making graph.png in: %s' % log_dir)
        cmd = ['dot', '-Tpng', '%s' % dot_file, '-o', '%s' % png_file]
        subprocess.call(cmd)


if __name__ == '__main__':
    dot_to_png()
