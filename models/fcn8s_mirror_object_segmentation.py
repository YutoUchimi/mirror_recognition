import os.path as osp

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import fcn
import numpy as np


class FCN8sMirrorObjectSegmentation(chainer.Chain):

    pretrained_model = osp.expanduser(
        '~/data/models/chainer/fcn8s-atonce_from_caffe.npz')

    def __init__(self, n_class_object):
        n_class_object = n_class_object
        self.n_class_object = n_class_object
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super(FCN8sMirrorObjectSegmentation, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 100, **kwargs)
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

            self.score_fr_mirror = L.Convolution2D(4096, 2, 1, 1, 0, **kwargs)
            self.upscore2_mirror = L.Deconvolution2D(
                2, 2, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore8_mirror = L.Deconvolution2D(
                2, 2, 16, 8, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.score_pool3_mirror = L.Convolution2D(
                256, 2, 1, 1, 0, **kwargs)
            self.score_pool4_mirror = L.Convolution2D(
                512, 2, 1, 1, 0, **kwargs)
            self.upscore_pool4_mirror = L.Deconvolution2D(
                2, 2, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

            self.score_fr_object = L.Convolution2D(
                4096, n_class_object, 1, 1, 0, **kwargs)
            self.upscore2_object = L.Deconvolution2D(
                n_class_object, n_class_object, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore8_object = L.Deconvolution2D(
                n_class_object, n_class_object, 16, 8, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.score_pool3_object = L.Convolution2D(
                256, n_class_object, 1, 1, 0, **kwargs)
            self.score_pool4_object = L.Convolution2D(
                512, n_class_object, 1, 1, 0, **kwargs)
            self.upscore_pool4_object = L.Deconvolution2D(
                n_class_object, n_class_object, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def segment_mirror(self, bgr):
        h = F.relu(self.conv1_1(bgr))
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

        h = self.score_fr_mirror(fc7)
        score_fr_mirror = h  # 1/32

        h = 0.01 * pool4
        h = self.score_pool4_mirror(h)
        score_pool4_mirror = h  # 1/16

        h = self.upscore2_mirror(score_fr_mirror)
        upscore2_mirror = h  # 1/16

        h = score_pool4_mirror[
            :, :,
            5:5 + upscore2_mirror.data.shape[2],
            5:5 + upscore2_mirror.data.shape[3]]
        score_pool4c_mirror = h  # 1/16

        h = upscore2_mirror + score_pool4c_mirror
        fuse_pool4_mirror = h  # 1/16

        h = self.upscore_pool4_mirror(fuse_pool4_mirror)
        upscore_pool4_mirror = h  # 1/8

        h = 0.0001 * pool3
        h = self.score_pool3_mirror(h)
        score_pool3_mirror = h  # 1/8

        h = score_pool3_mirror[
            :, :,
            9:9 + upscore_pool4_mirror.data.shape[2],
            9:9 + upscore_pool4_mirror.data.shape[3]]
        score_pool3c_mirror = h  # 1/8

        h = upscore_pool4_mirror + score_pool3c_mirror
        fuse_pool3_mirror = h  # 1/8

        h = self.upscore8_mirror(fuse_pool3_mirror)
        upscore8_mirror = h  # 1/1

        h = upscore8_mirror[
            :, :,
            31:31 + bgr.shape[2], 31:31 + bgr.shape[3]]
        score_mirror = h  # 1/1

        return score_mirror, pool3, pool4, fc7

    def segment_object(self, bgr, pool3, pool4, fc7):
        h = self.score_fr_object(fc7)
        score_fr_object = h  # 1/32

        h = 0.01 * pool4
        h = self.score_pool4_object(h)
        score_pool4_object = h  # 1/16

        h = self.upscore2_object(score_fr_object)
        upscore2_object = h  # 1/16

        h = score_pool4_object[
            :, :,
            5:5 + upscore2_object.data.shape[2],
            5:5 + upscore2_object.data.shape[3]]
        score_pool4c_object = h  # 1/16

        h = upscore2_object + score_pool4c_object
        fuse_pool4_object = h  # 1/16

        h = self.upscore_pool4_object(fuse_pool4_object)
        upscore_pool4_object = h  # 1/8

        h = 0.0001 * pool3
        h = self.score_pool3_object(h)
        score_pool3_object = h  # 1/8

        h = score_pool3_object[
            :, :,
            9:9 + upscore_pool4_object.data.shape[2],
            9:9 + upscore_pool4_object.data.shape[3]]
        score_pool3c_object = h  # 1/8

        h = upscore_pool4_object + score_pool3c_object
        fuse_pool3_object = h  # 1/8

        h = self.upscore8_object(fuse_pool3_object)
        upscore8_object = h  # 1/1

        h = upscore8_object[
            :, :,
            31:31 + bgr.shape[2],
            31:31 + bgr.shape[3]]
        score_object = h  # 1/1

        return score_object

    def compute_seg_loss(self, score, gt):
        seg_loss = F.softmax_cross_entropy(
            score, gt, normalize=True)
        return seg_loss

    def compute_loss(self, score_mirror, score_object, gt_mirror, gt_object):
        loss_mirror = self.compute_seg_loss(score_mirror, gt_mirror)
        loss_object = self.compute_seg_loss(score_object, gt_object)

        # XXX: What is proper loss function?
        coef = [1, 1]
        loss = (coef[0] * loss_mirror +
                coef[1] * loss_object)
        if self.xp.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')

        batch_size = len(score_mirror)
        assert batch_size == 1

        # N, C, H, W -> C, H, W
        gt_mirror = cuda.to_cpu(gt_mirror)[0]
        gt_object = cuda.to_cpu(gt_object)[0]
        pred_mirror = cuda.to_cpu(F.argmax(score_mirror, axis=1).data)[0]
        pred_object = cuda.to_cpu(F.argmax(score_object, axis=1).data)[0]
        pred_mirror = pred_mirror.astype(np.int32)
        pred_object = pred_object.astype(np.int32)

        # Evaluate Mean IU
        mean_iu_mirror = fcn.utils.label_accuracy_score(
            [gt_mirror], [pred_mirror], n_class=2)[2]
        mean_iu_object = fcn.utils.label_accuracy_score(
            [gt_object], [pred_object], n_class=self.n_class_object)[2]

        chainer.reporter.report({
            'loss': loss,
            'loss_mirror': loss_mirror,
            'loss_object': loss_object,
            'mean_iu_mirror': mean_iu_mirror,
            'mean_iu_object': mean_iu_object,
        }, self)

        return loss

    def __call__(self, bgr, gt_mirror=None, gt_object=None):
        score_mirror, pool3, pool4, fc7 = self.segment_mirror(bgr)
        score_object = self.segment_object(bgr, pool3, pool4, fc7)
        self.score_mirror = score_mirror
        self.score_object = score_object

        if gt_mirror is None or gt_object is None:
            assert not chainer.config.train
            return

        loss = self.compute_loss(
            score_mirror, score_object, gt_mirror, gt_object)
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
            if l.name.startswith('conv'):
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
