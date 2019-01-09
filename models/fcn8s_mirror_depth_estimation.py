from __future__ import division

import os.path as osp

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import fcn
import numpy as np


class FCN8sMirrorDepthEstimation(chainer.Chain):

    # 0.5 < depth < 5.0
    min_depth = 0.5
    max_depth = 5.0

    pretrained_model = osp.expanduser(
        '~/data/models/chainer/fcn8s-atonce_from_caffe.npz')

    def __init__(self, n_class):
        n_class = n_class
        self.n_class = n_class
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(FCN8sMirrorDepthEstimation, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3 + 3, 64, 3, 1, 100, **kwargs)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, **kwargs)

            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, **kwargs)

            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)

            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.fc6_label = L.Convolution2D(512, 4096, 7, 1, 0, **kwargs)
            self.fc7_label = L.Convolution2D(4096, 4096, 1, 1, 0, **kwargs)

            self.score_fr_label = L.Convolution2D(
                4096, n_class, 1, 1, 0, **kwargs)
            self.upscore2_label = L.Deconvolution2D(
                n_class, n_class, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore8_label = L.Deconvolution2D(
                n_class, n_class, 16, 8, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.score_pool3_label = L.Convolution2D(
                256, n_class, 1, 1, 0, **kwargs)
            self.score_pool4_label = L.Convolution2D(
                512, n_class, 1, 1, 0, **kwargs)
            self.upscore_pool4_label = L.Deconvolution2D(
                n_class, n_class, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

            self.fc6_depth = L.Convolution2D(512, 4096, 7, 1, 0, **kwargs)
            self.fc7_depth = L.Convolution2D(4096, 4096, 1, 1, 0, **kwargs)

            self.score_fr_depth = L.Convolution2D(4096, 1, 1, 1, 0, **kwargs)
            self.upscore2_depth = L.Deconvolution2D(
                1, 1, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore8_depth = L.Deconvolution2D(
                1, 1, 16, 8, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.score_pool3_depth = L.Convolution2D(256, 1, 1, 1, 0, **kwargs)
            self.score_pool4_depth = L.Convolution2D(512, 1, 1, 1, 0, **kwargs)
            self.upscore_pool4_depth = L.Deconvolution2D(
                1, 1, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def predict_label(self, bgr, depth_bgr):
        h = F.concat((bgr, depth_bgr), axis=1)

        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool1 = h  # 1/2

        h = F.relu(self.conv2_1(pool1))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool2 = h  # 1/4

        h = F.relu(self.conv3_1(pool2))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool3 = h  # 1/8

        h = F.relu(self.conv4_1(pool3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool4 = h  # 1/16

        h = F.relu(self.conv5_1(pool4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool5 = h  # 1/32

        h = F.relu(self.fc6_label(pool5))
        h = F.dropout(h, ratio=0.5)
        fc6_label = h  # 1/32

        h = F.relu(self.fc7_label(fc6_label))
        h = F.dropout(h, ratio=0.5)
        fc7_label = h  # 1/32

        h = self.score_fr_label(fc7_label)
        score_fr_label = h  # 1/32

        h = 0.01 * pool4
        h = self.score_pool4_label(h)
        score_pool4_label = h  # 1/16

        h = self.upscore2_label(score_fr_label)
        upscore2_label = h  # 1/16

        h = score_pool4_label[
            :, :,
            5:5 + upscore2_label.data.shape[2],
            5:5 + upscore2_label.data.shape[3]
        ]
        score_pool4c_label = h  # 1/16

        h = upscore2_label + score_pool4c_label
        fuse_pool4_label = h  # 1/16

        h = self.upscore_pool4_label(fuse_pool4_label)
        upscore_pool4_label = h  # 1/8

        h = 0.0001 * pool3
        h = self.score_pool3_label(h)
        score_pool3_label = h  # 1/8

        h = score_pool3_label[
            :, :,
            9:9 + upscore_pool4_label.data.shape[2],
            9:9 + upscore_pool4_label.data.shape[3]
        ]
        score_pool3c_label = h  # 1/8

        h = upscore_pool4_label + score_pool3c_label
        fuse_pool3_label = h  # 1/8

        h = self.upscore8_label(fuse_pool3_label)
        upscore8_label = h  # 1/1

        h = upscore8_label[:, :, 31:31 + bgr.shape[2], 31:31 + bgr.shape[3]]
        score_label = h  # 1/1

        return score_label, pool3, pool4, pool5

    def predict_depth(self, depth_bgr, pool3, pool4, pool5):
        h = F.relu(self.fc6_depth(pool5))
        h = F.dropout(h, ratio=0.5)
        fc6_depth = h  # 1/32

        h = F.relu(self.fc7_depth(fc6_depth))
        h = F.dropout(h, ratio=0.5)
        fc7_depth = h  # 1/32

        h = self.score_fr_depth(fc7_depth)
        score_fr_depth = h  # 1/32

        h = 0.01 * pool4
        h = self.score_pool4_depth(h)
        score_pool4_depth = h  # 1/16

        h = self.upscore2_depth(score_fr_depth)
        upscore2_depth = h  # 1/16

        h = score_pool4_depth[
            :, :,
            5:5 + upscore2_depth.data.shape[2],
            5:5 + upscore2_depth.data.shape[3]
        ]
        score_pool4c_depth = h  # 1/16

        h = upscore2_depth + score_pool4c_depth
        fuse_pool4_depth = h  # 1/16

        h = self.upscore_pool4_depth(fuse_pool4_depth)
        upscore_pool4_depth = h  # 1/8

        h = 0.0001 * pool3
        h = self.score_pool3_depth(h)
        score_pool3_depth = h  # 1/8

        h = score_pool3_depth[
            :, :,
            9:9 + upscore_pool4_depth.data.shape[2],
            9:9 + upscore_pool4_depth.data.shape[3]
        ]
        score_pool3c_depth = h  # 1/8

        h = upscore_pool4_depth + score_pool3c_depth
        fuse_pool3_depth = h  # 1/8

        h = self.upscore8_depth(fuse_pool3_depth)
        upscore8_depth = h  # 1/1

        h = upscore8_depth[
            :, :,
            31:31 + depth_bgr.shape[2],
            31:31 + depth_bgr.shape[3]
        ]
        score_depth = h  # 1/1

        return score_depth

    def compute_loss_label(self, score_label, label_gt):
        # Segmentation loss
        seg_loss = F.softmax_cross_entropy(
            score_label, label_gt, normalize=True)

        return seg_loss

    def compute_loss_depth(self, score_depth, label_gt, depth_gt):
        assert label_gt.dtype == self.xp.int32

        # (-inf, inf) -> (0, 1) -> (min_depth, max_depth)
        h = F.sigmoid(score_depth)
        depth_pred = h * (self.max_depth - self.min_depth) + self.min_depth

        keep_regardless_mask = ~self.xp.isnan(depth_gt)
        if self.xp.sum(keep_regardless_mask) == 0:
            depth_loss_regardless_mask = 0
        else:
            depth_loss_regardless_mask = F.mean_squared_error(
                depth_pred[keep_regardless_mask],
                depth_gt[keep_regardless_mask])

        keep_only_mask = self.xp.logical_and(
            label_gt > 0, ~self.xp.isnan(depth_gt))
        if self.xp.sum(keep_only_mask) == 0:
            depth_loss_only_mask = 0
        else:
            depth_loss_only_mask = F.mean_squared_error(
                depth_pred[keep_only_mask],
                depth_gt[keep_only_mask])

        # Regression loss
        # XXX: What is proper loss function?
        coef = [1, 10]
        reg_loss = (coef[0] * depth_loss_regardless_mask +
                    coef[1] * depth_loss_only_mask)

        return reg_loss

    def compute_loss(self, score_label, score_depth, label_gt, depth_gt):
        seg_loss = self.compute_loss_label(score_label, label_gt)
        reg_loss = self.compute_loss_depth(score_depth, depth_gt)

        # Loss
        # XXX: What is proper loss function?
        coef = [1, 10]
        loss = coef[0] * seg_loss + coef[1] * reg_loss
        if self.xp.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')

        batch_size = len(score_label)
        assert batch_size == 1

        # GPU -> CPU
        # N, C, H, W -> C, H, W
        label_gt = cuda.to_cpu(label_gt)[0]
        label_pred = cuda.to_cpu(F.argmax(score_label, axis=1).data)[0]
        depth_gt = cuda.to_cpu(depth_gt)[0]
        h = F.sigmoid(score_depth)
        depth_pred = h * (self.max_depth - self.min_depth) + self.min_depth
        depth_pred = cuda.to_cpu(depth_pred)[0]

        # Evaluate Mean IU
        miou = fcn.utils.label_accuracy_score(
            [label_gt], [label_pred], n_class=self.n_class)[2]

        # Evaluate Depth Accuracy
        depth_acc = {}
        for thresh in [0.01, 0.03, 0.10, 0.30, 1.00]:
            t_lbl_fg = label_gt > 0
            p_lbl_fg = label_pred > 0
            if np.sum(t_lbl_fg) == 0:
                acc = np.nan
            else:
                # {TP and (|error| < thresh)} / (TP or FP or FN)
                depth_gt_cp = np.copy(depth_gt)
                depth_gt_cp[np.isnan(depth_gt_cp)] = np.inf
                numer = np.sum(
                    np.logical_and(
                        np.logical_and(t_lbl_fg, p_lbl_fg),
                        np.abs(depth_gt_cp - depth_pred) < thresh)
                )
                denom = np.sum(np.logical_or(t_lbl_fg, p_lbl_fg))
                acc = 1.0 * numer / denom
            depth_acc['%.2f' % thresh] = acc

        chainer.reporter.report({
            'loss': loss,
            'seg_loss': seg_loss,
            'reg_loss': reg_loss,
            'miou': miou,
            'depth_acc<0.01': depth_acc['0.01'],
            'depth_acc<0.03': depth_acc['0.03'],
            'depth_acc<0.10': depth_acc['0.10'],
            'depth_acc<0.30': depth_acc['0.30'],
            'depth_acc<1.00': depth_acc['1.00'],
        }, self)

        return loss

    def __call__(self, bgr, depth_bgr, label_gt=None, depth_gt=None):
        score_label, pool3, pool4, fc7 = self.predict_label(bgr, depth_bgr)
        score_depth = self.predict_depth(depth_bgr, pool3, pool4, fc7)
        self.score_label = score_label
        self.score_depth = score_depth

        if label_gt is None or depth_gt is None:
            assert not chainer.config.train
            return

        loss = self.compute_loss(score_label, score_depth, label_gt, depth_gt)

        return loss

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='https://drive.google.com/uc?id=0B9P1L--7Wd2vZ1RJdXotZkNhSEk',
            path=cls.pretrained_model,
            md5='5f3ffdc7fae1066606e1ef45cfda548f',
        )

    def init_from_vgg16(self, vgg16):
        for l in self.children():
            if l.name == 'conv1_1':
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.shape[0] == l2.W.shape[0]
                assert l1.W.shape[1] * 2 == l2.W.shape[1]
                assert l1.W.shape[2:] == l2.W.shape[2:]
                assert l1.b.shape == l2.b.shape
                l2.W.data[:, :l1.W.shape[1], :, :] = l1.W.data[...]
                l2.W.data[:, l1.W.shape[1]:, :, :] = l1.W.data[...]
                l2.b.data[...] = l1.b.data[...]
            elif l.name.startswith('conv'):
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.shape == l2.W.shape
                assert l1.b.shape == l2.b.shape
                l2.W.data[...] = l1.W.data[...]
                l2.b.data[...] = l1.b.data[...]
            elif l.name.startswith('fc'):
                l1 = getattr(vgg16, l.name.split('_')[0])
                l2 = getattr(self, l.name)
                assert l1.W.size == l2.W.size
                assert l1.b.size == l2.b.size
                l2.W.data[...] = l1.W.data.reshape(l2.W.shape)[...]
                l2.b.data[...] = l1.b.data.reshape(l2.b.shape)[...]
