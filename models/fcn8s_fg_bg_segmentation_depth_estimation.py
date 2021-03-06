import os.path as osp

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import fcn
import numpy as np


class FCN8sMirrorDepthEstimation(chainer.Chain):

    # 0.2 < depth < 3.0
    min_depth = 0.2
    max_depth = 3.0

    pretrained_model = osp.expanduser(
        '~/data/models/chainer/fcn8s-atonce_from_caffe.npz')

    def __init__(self, n_class_including_mirror):
        n_class = n_class_including_mirror
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

            self.fc6 = L.Convolution2D(512, 4096, 7, 1, 0, **kwargs)
            self.fc7 = L.Convolution2D(4096, 4096, 1, 1, 0, **kwargs)

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

            self.score_fr_depth = L.Convolution2D(
                4096, 2, 1, 1, 0, **kwargs)

            self.upscore2_depth = L.Deconvolution2D(
                2, 2, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())
            self.upscore8_depth = L.Deconvolution2D(
                2, 2, 16, 8, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

            self.score_pool3_depth = L.Convolution2D(256, 2, 1, 1, 0, **kwargs)
            self.score_pool4_depth = L.Convolution2D(512, 2, 1, 1, 0, **kwargs)

            self.upscore_pool4_depth = L.Deconvolution2D(
                2, 2, 4, 2, 0, nobias=True,
                initialW=fcn.initializers.UpsamplingDeconvWeight())

    def predict_mask(self, rgb, depth_viz):
        h = F.concat((rgb, depth_viz), axis=1)

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

        h = self.score_fr_label(fc7)
        score_fr_label = h  # 1/32

        h = 0.01 * pool4
        h = self.score_pool4_label(h)
        score_pool4_label = h  # 1/16

        h = self.upscore2_label(score_fr_label)
        upscore2_label = h  # 1/16

        h = score_pool4_label[:, :,
                              5:5 + upscore2_label.data.shape[2],
                              5:5 + upscore2_label.data.shape[3]]
        score_pool4c_label = h  # 1/16

        h = upscore2_label + score_pool4c_label
        fuse_pool4_label = h  # 1/16

        h = self.upscore_pool4_label(fuse_pool4_label)
        upscore_pool4_label = h  # 1/8

        h = 0.0001 * pool3
        h = self.score_pool3_label(h)
        score_pool3_label = h  # 1/8

        h = score_pool3_label[:, :,
                              9:9 + upscore_pool4_label.data.shape[2],
                              9:9 + upscore_pool4_label.data.shape[3]]
        score_pool3c_label = h  # 1/8

        h = upscore_pool4_label + score_pool3c_label
        fuse_pool3_label = h  # 1/8

        h = self.upscore8_label(fuse_pool3_label)
        upscore8_label = h  # 1/1

        h = upscore8_label[:, :,
                           31:31 + rgb.shape[2], 31:31 + rgb.shape[3]]
        score_label = h  # 1/1

        return score_label, pool3, pool4, fc7

    def predict_depth(self, depth_viz, pool3, pool4, fc7):
        h = self.score_fr_depth(fc7)
        score_fr_depth = h  # 1/32

        h = 0.01 * pool4
        h = self.score_pool4_depth(h)
        score_pool4_depth = h  # 1/16

        h = self.upscore2_depth(score_fr_depth)
        upscore2_depth = h  # 1/16

        h = score_pool4_depth[:, :,
                              5:5 + upscore2_depth.data.shape[2],
                              5:5 + upscore2_depth.data.shape[3]]
        score_pool4c_depth = h  # 1/16

        h = upscore2_depth + score_pool4c_depth
        fuse_pool4_depth = h  # 1/16

        h = self.upscore_pool4_depth(fuse_pool4_depth)
        upscore_pool4_depth = h  # 1/8

        h = 0.0001 * pool3
        h = self.score_pool3_depth(h)
        score_pool3_depth = h  # 1/8

        h = score_pool3_depth[:, :,
                              9:9 + upscore_pool4_depth.data.shape[2],
                              9:9 + upscore_pool4_depth.data.shape[3]]
        score_pool3c_depth = h  # 1/8

        h = upscore_pool4_depth + score_pool3c_depth
        fuse_pool3_depth = h  # 1/8

        h = self.upscore8_depth(fuse_pool3_depth)
        upscore8_depth = h  # 1/1

        h = upscore8_depth[:, :,
                           31:31 + depth_viz.shape[2],
                           31:31 + depth_viz.shape[3]]
        score_depth = h  # 1/1

        return score_depth

    def compute_loss_label(self, score_label, label_gt):
        # segmentation loss
        seg_loss = F.softmax_cross_entropy(
            score_label, label_gt, normalize=True)

        return seg_loss

    def compute_loss_depth(self, score_depth, label_gt, depth_gt):
        assert label_gt.dtype == self.xp.int32

        # regression loss
        h = F.sigmoid(score_depth)  # (0, 1)
        depth_pred = h * (self.max_depth - self.min_depth)
        depth_pred += self.min_depth

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

        # XXX: What is proper loss function?
        coef = [1, 10]
        reg_loss = (coef[0] * depth_loss_regardless_mask +
                    coef[1] * depth_loss_only_mask)

        return reg_loss

    def compute_loss(self, score_label, score_depth,
                     label_gt_fg, label_gt_bg, depth_gt_fg, depth_gt_bg):
        score_label_fg = score_label[:, :self.n_class, :, :]
        score_label_bg = score_label[:, self.n_class:, :, :]
        # N, 2, H, W -> N, 1, H, W
        score_depth_fg = score_depth[:, 0, :, :][:, self.xp.newaxis, :, :]
        score_depth_bg = score_depth[:, 1, :, :][:, self.xp.newaxis, :, :]

        seg_loss_fg = self.compute_loss_label(score_label_fg, label_gt_fg)
        seg_loss_bg = self.compute_loss_label(score_label_bg, label_gt_bg)
        reg_loss_fg = self.compute_loss_depth(score_depth_fg, depth_gt_fg)
        reg_loss_bg = self.compute_loss_depth(score_depth_bg, depth_gt_bg)

        # XXX: What is proper loss function?
        coef = [1, 1, 10, 10]
        loss = (coef[0] * seg_loss_fg +
                coef[1] * seg_loss_bg +
                coef[2] * reg_loss_fg +
                coef[3] * reg_loss_bg)
        if self.xp.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')

        batch_size = len(score_label_fg)
        assert batch_size == 1

        # N, C, H, W -> C, H, W
        label_gt_fg = cuda.to_cpu(label_gt_fg)[0]
        label_gt_bg = cuda.to_cpu(label_gt_bg)[0]
        label_pred_fg = cuda.to_cpu(F.argmax(score_label_fg, axis=1).data)[0]
        label_pred_bg = cuda.to_cpu(F.argmax(score_label_bg, axis=1).data)[0]
        depth_gt_fg = cuda.to_cpu(depth_gt_fg)[0]
        depth_gt_bg = cuda.to_cpu(depth_gt_bg)[0]

        h = F.sigmoid(score_depth)[:, 0, :, :]
        depth_pred_fg = h * (self.max_depth - self.min_depth) + self.min_depth
        depth_pred_fg = cuda.to_cpu(depth_pred_fg)[0]

        h = F.sigmoid(score_depth)[:, 1, :, :]
        depth_pred_bg = h * (self.max_depth - self.min_depth) + self.min_depth
        depth_pred_bg = cuda.to_cpu(depth_pred_bg)[0]

        # Evaluate Mean IU
        mean_iu_fg = fcn.utils.label_accuracy_score(
            [label_gt_fg], [label_pred_fg], n_class=self.n_class)[2]
        mean_iu_bg = fcn.utils.label_accuracy_score(
            [label_gt_bg], [label_pred_bg], n_class=self.n_class)[2]

        # Evaluate Depth Accuracy
        depth_acc_fg = {}
        depth_acc_bg = {}
        for ground in ['fg', 'bg']:
            if ground == 'fg':
                label_gt = label_gt_fg
                label_pred = label_pred_fg
                depth_gt = depth_gt_fg
                depth_pred = depth_pred_fg
            else:
                label_gt = label_gt_bg
                label_pred = label_pred_bg
                depth_gt = depth_gt_bg
                depth_pred = depth_pred_bg
            for thresh in [0.01, 0.02, 0.05]:
                t_lbl_fg = label_gt > 0
                p_lbl_fg = label_pred > 0
                if np.sum(t_lbl_fg) == 0:
                    acc = np.nan
                else:
                    depth_gt_cp = np.copy(depth_gt)
                    depth_gt_cp[np.isnan(depth_gt_cp)] = np.inf
                    numer = np.sum(np.logical_and(
                        np.logical_and(t_lbl_fg, p_lbl_fg),
                        np.abs(depth_gt_cp - depth_pred) < thresh))
                    denom = np.sum(np.logical_or(t_lbl_fg, p_lbl_fg))
                    acc = 1.0 * numer / denom
                if ground == 'fg':
                    depth_acc_fg['%.2f' % thresh] = acc
                elif ground == 'bg':
                    depth_acc_bg['%.2f' % thresh] = acc

        chainer.reporter.report({
            'loss': loss,
            'seg_loss_fg': seg_loss_fg,
            'seg_loss_bg': seg_loss_bg,
            'reg_loss_fg': reg_loss_fg,
            'reg_loss_bg': reg_loss_bg,
            'mean_iu_fg': mean_iu_fg,
            'mean_iu_bg': mean_iu_bg,
            'depth_acc_fg<0.01': depth_acc_fg['0.01'],
            'depth_acc_fg<0.02': depth_acc_fg['0.02'],
            'depth_acc_fg<0.05': depth_acc_fg['0.05'],
            'depth_acc_bg<0.01': depth_acc_bg['0.01'],
            'depth_acc_bg<0.02': depth_acc_bg['0.02'],
            'depth_acc_bg<0.05': depth_acc_bg['0.05'],
        }, self)

        return loss

    def __call__(self, rgb, depth_viz, label_gt_fg=None, label_gt_bg=None,
                 depth_gt_fg=None, depth_gt_bg=None):
        score_label, pool3, pool4, fc7 = self.predict_mask(rgb)
        score_depth = self.predict_depth(depth_viz, pool3, pool4, fc7)
        self.score_label = score_label
        self.score_depth = score_depth

        if label_gt_fg is None or label_gt_bg is None or \
           depth_gt_fg is None or depth_gt_bg is None:
            assert not chainer.config.train
            return

        loss = self.compute_loss(
            score_label, score_depth,
            label_gt_fg, label_gt_bg, depth_gt_fg, depth_gt_bg)

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
                assert l1.W.size * 2 == l2.W.shape
                assert l1.b.size == l2.b.size
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
            elif l.name in ['fc6', 'fc7']:
                l1 = getattr(vgg16, l.name)
                l2 = getattr(self, l.name)
                assert l1.W.size == l2.W.size
                assert l1.b.size == l2.b.size
                l2.W.data[...] = l1.W.data.reshape(l2.W.shape)[...]
                l2.b.data[...] = l1.b.data.reshape(l2.b.shape)[...]
