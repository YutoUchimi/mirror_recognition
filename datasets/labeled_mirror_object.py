import os
import os.path as osp

import chainer
import cv2
import numpy as np
import PIL.Image
import skimage.io

import imgaug.augmenters as iaa
import mvtk


class LabeledMirrorObjectDataset(chainer.dataset.DatasetMixin):

    # TODO(unknown): Get class names from dataset dir.
    mirror_class_names = np.array([
        '_background_',
        'mirror',
        ], dtype=np.str)
    object_class_names = np.array([
        '_background_',
        '_tote_',
        'band_aid_tape',
        'bath_sponge',
        'burts_bees_baby_wipes',
        'colgate_toothbrush_4pk',
        'crayons',
        'duct_tape',
        'expo_eraser',
        'hanes_socks',
        'laugh_out_loud_jokes',
        'tissue_box',
    ], dtype=np.str)
    mirror_class_names.setflags(write=0)
    object_class_names.setflags(write=0)

    _files = set([
        'image.png',
        'mirror_label.png',
        'mirror_label_names.txt',
        'mirror_label_viz.png',
        'object_label.png',
        'object_label_names.txt',
        'object_label_viz.png',
        ])

    root_dir = osp.expanduser(
        '~/data/mvtk/mirror_recognition/labeled_mirror_object_dataset')
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, split, aug=False):
        assert split in ['train', 'test']
        self.split = split
        self.aug = aug

        self._files_dirs = []
        for date_dir in sorted(os.listdir(self.root_dir)):
            date_dir = osp.join(self.root_dir, date_dir)
            split_dir = osp.join(date_dir, split)
            for files_dir in sorted(os.listdir(split_dir)):
                files_dir = osp.join(split_dir, files_dir)
                files = set(os.listdir(files_dir))
                assert self._files.issubset(files), (
                    'In {}: File set does not match.\n'
                    'Expected: {}\nActual: {}'
                    .format(files_dir, self._files, files))
                self._files_dirs.append(files_dir)

    def __len__(self):
        return len(self._files_dirs)

    def get_example(self, i):
        files_dir = self._files_dirs[i]

        image_file = osp.join(files_dir, 'image.png')
        image = skimage.io.imread(image_file)
        assert image.dtype == np.uint8
        assert image.ndim == 3

        mirror_label_file = osp.join(files_dir, 'mirror_label.png')
        with open(mirror_label_file, 'r') as f:
            mirror_label = np.asarray(PIL.Image.open(f)).astype(np.int32)
        assert mirror_label.dtype == np.int32
        assert mirror_label.ndim == 2

        object_label_file = osp.join(files_dir, 'object_label.png')
        with open(object_label_file, 'r') as f:
            object_label = np.asarray(PIL.Image.open(f)).astype(np.int32)
        assert object_label.dtype == np.int32
        assert object_label.ndim == 2

        # Data augmentation
        if self.aug:
            # 1. Color augmentation
            obj_datum = dict(img=image)
            random_state = np.random.RandomState()

            def st(x):
                return iaa.Sometimes(0.3, x)

            augs = [
                st(iaa.Add([-50, 50], per_channel=True)),
                st(iaa.InColorspace(
                    'HSV', children=iaa.WithChannels(
                        [1, 2], iaa.Multiply([0.5, 2])))),
                st(iaa.GaussianBlur(sigma=[0.0, 1.0])),
                st(iaa.AdditiveGaussianNoise(
                    scale=(0.0, 0.1 * 255), per_channel=True)),
            ]
            obj_datum = next(mvtk.aug.augment_object_data(
                [obj_datum], random_state=random_state, augmentations=augs))
            image = obj_datum['img']

            # 2. Geometric augmentation
            np.random.seed()
            if np.random.uniform() < 0.5:
                image = np.fliplr(image)
                mirror_label = np.fliplr(mirror_label)
                object_label = np.fliplr(object_label)
            if np.random.uniform() < 0.5:
                image = np.flipud(image)
                mirror_label = np.flipud(mirror_label)
                object_label = np.flipud(object_label)
            if np.random.uniform() < 0.5:
                angle = (np.random.uniform() * 180) - 90
                image = self.rotate_image(image, angle, cv2.INTER_LINEAR)
                mirror_label = self.rotate_image(
                    mirror_label, angle, cv2.INTER_NEAREST)
                object_label = self.rotate_image(
                    object_label, angle, cv2.INTER_NEAREST)
        return image, mirror_label, object_label

    def rotate_image(self, in_img, angle, flags=cv2.INTER_LINEAR):
        rot_mat = cv2.getRotationMatrix2D(
            center=(in_img.shape[1] / 2, in_img.shape[0] / 2),
            angle=angle, scale=1)
        rot_img = cv2.warpAffine(
            src=in_img, M=rot_mat,
            dsize=(in_img.shape[1], in_img.shape[0]), flags=flags)
        return rot_img

    # def rotate_depth_image(self, in_img, angle, flags=cv2.INTER_LINEAR):
    #     rot_mat = cv2.getRotationMatrix2D(
    #         center=(in_img.shape[1] / 2, in_img.shape[0] / 2),
    #         angle=angle, scale=1)
    #     ones = np.ones(in_img.shape, dtype=np.int32)
    #     rot_keep = cv2.warpAffine(
    #         src=ones, M=rot_mat,
    #         dsize=(in_img.shape[1], in_img.shape[0]),
    #         flags=cv2.INTER_NEAREST)
    #     rot_keep = rot_keep.astype(np.bool)
    #     rot_img = cv2.warpAffine(
    #         src=in_img, M=rot_mat,
    #         dsize=(in_img.shape[1], in_img.shape[0]), flags=flags)
    #     rot_img[rot_keep == False] = np.nan  # NOQA
    #     return rot_img

    def visualize(self, index):
        image, mirror_label, object_label = self[index]

        print('[%04d] %s' % (index, '>' * 75))
        print('image_shape: %s' % repr(image.shape))
        print('[%04d] %s' % (index, '<' * 75))
        mirror_label = mvtk.image.label2rgb(
            mirror_label.astype(np.int32), img=image,
            label_names=self.mirror_class_names, alpha=0.7)
        object_label = mvtk.image.label2rgb(
            object_label.astype(np.int32), img=image,
            label_names=self.object_class_names, alpha=0.7)
        viz = mvtk.image.tile(
            [image, mirror_label, object_label], (1, 3))
        return mvtk.image.resize(viz, size=600 * 600)  # for small window
