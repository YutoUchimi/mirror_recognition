import os.path as osp

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import fcn
import numpy as np


class FCN8sMirrorSegmentationWithDepth(chainer.Chain):

    pretrained_model = osp.expanduser(
        '~/data/models/chainer/fcn8s-atonce_from_caffe.npz')

    def __init__(self, n_class):
        n_class = n_class
        self.n_class = n_class
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(FCN8sMirrorSegmentationWithDepth, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3 + 1, 64, 3, 1, 100, **kwargs)
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

            self.fc6 = L.Convolution2D(512, 4096, 7, 1, 0, **kwargs)
            self.fc7 = L.Convolution2D(4096, 4096, 1, 1, 0, **kwargs)

            self.score_fr = L.Convolution2D(4096, n_class, 1, 1, 0, **kwargs)
            self.upscore2 = L.Deconvolution2D(
                n_class, n_class, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore8 = L.Deconvolution2D(
                n_class, n_class, 16, 8, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.score_pool3 = L.Convolution2D(256, n_class, 1, 1, 0, **kwargs)
            self.score_pool4 = L.Convolution2D(512, n_class, 1, 1, 0, **kwargs)
            self.upscore_pool4 = L.Deconvolution2D(
                n_class, n_class, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def segment(self, bgr, depth):
        h = F.concat((bgr, depth), axis=1)
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

        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=0.5)
        fc6 = h  # 1/32

        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=0.5)
        fc7 = h  # 1/32

        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        h = 0.01 * pool4
        h = self.score_pool4(h)
        score_pool4 = h  # 1/16

        h = self.upscore2(score_fr)
        upscore2 = h  # 1/16

        h = score_pool4[
            :, :,
            5:5 + upscore2.data.shape[2],
            5:5 + upscore2.data.shape[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c
        fuse_pool4 = h  # 1/16

        h = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = h  # 1/8

        h = 0.0001 * pool3
        h = self.score_pool3(h)
        score_pool3 = h  # 1/8

        h = score_pool3[
            :, :,
            9:9 + upscore_pool4.data.shape[2],
            9:9 + upscore_pool4.data.shape[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c
        fuse_pool3 = h  # 1/8

        h = self.upscore8(fuse_pool3)
        upscore8 = h  # 1/1

        h = upscore8[
            :, :,
            31:31 + bgr.shape[2], 31:31 + bgr.shape[3]]
        score = h  # 1/1

        return score, pool3, pool4, fc7

    def compute_seg_loss(self, score, gt):
        seg_loss = F.softmax_cross_entropy(score, gt, normalize=True)
        return seg_loss

    def compute_loss(self, score, gt):
        loss = self.compute_seg_loss(score, gt)
        if self.xp.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')

        batch_size = len(score)
        assert batch_size == 1

        # N, C, H, W -> C, H, W
        gt = cuda.to_cpu(gt)[0]
        pred = cuda.to_cpu(F.argmax(score, axis=1).data)[0]
        pred = pred.astype(np.int32)

        # Evaluate Mean IU
        miou = fcn.utils.label_accuracy_score([gt], [pred], n_class=2)[2]

        chainer.reporter.report({
            'loss': loss,
            'miou': miou,
        }, self)

        return loss

    def __call__(self, bgr, depth, gt=None):
        score, pool3, pool4, fc7 = self.segment(bgr, depth)
        self.score = score

        if gt is None:
            assert not chainer.config.train
            return

        loss = self.compute_loss(score, gt)
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
                assert l1.W.shape[1] == l2.W.shape[1] - 1
                assert l1.W.shape[2:] == l2.W.shape[2:]
                assert l1.b.shape == l2.b.shape
                l2.W.data[:, :l2.W.shape[1], :, :] = l1.W.data[:, :, :, :]
                mean_W = np.mean(l1.W.data, axis=1)[:, np.newaxis, :, :]
                l2.W.data[:, l2.W.shape[1], :, :] = mean_W
            elif l.name.startswith('conv'):
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.shape == l2.W.shape
                assert l1.b.shape == l2.b.shape
                l2.W.data[...] = l1.W.data[...]
                l2.b.data[...] = l1.b.data[...]
            elif l.name in ['fc6', 'fc7']:
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.size == l2.W.size
                assert l1.b.size == l2.b.size
                l2.W.data[...] = l1.W.data.reshape(l2.W.shape)[...]
                l2.b.data[...] = l1.b.data.reshape(l2.b.shape)[...]
