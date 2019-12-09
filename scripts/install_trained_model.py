#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'mirror_recognition'

    download_data(
        pkg_name=PKG,
        path='trained_model/fcn_mirror_depth_prediction_20191206.tar.gz',
        url='https://drive.google.com/uc?id=1x_axX_r6Pwfl2NSqCB7bTenkfbDzGvFj',
        md5='2f4fd835fd11240fcca65a9d4711e96f',
        extract=True
    )


if __name__ == '__main__':
    main()
