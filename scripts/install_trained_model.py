#!/usr/bin/env python

from jsk_data import download_data


def main():
    PKG = 'mirror_recognition'

    download_data(
        pkg_name=PKG,
        path='trained_model/fcn_mirror_depth_prediction_20190130.tgz',
        url='https://drive.google.com/uc?id=1jV2mujNpYIaoOV6-KNctzPLAIi4ZQJpD',
        md5='52228c0123b349ffd80356f84925aeea',
        extract=True
    )


if __name__ == '__main__':
    main()
