#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'mirror_recognition'

    download_data(
        pkg_name=PKG,
        path='trained_model/fcn_mirror_depth_prediction_20191205.tar.gz',
        url='https://drive.google.com/uc?id=19IgX3qOzU_ZAS5sdHzsp8y34JfxzR50-',
        md5='57d96f65dcd58a375ff1bf3308d3ddb9',
        extract=True
    )


if __name__ == '__main__':
    main()
