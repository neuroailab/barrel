'''
Copyright (C) 2014 New York University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import os
import time
import numpy as np
import ipdb

import theano
import theano.tensor as T

from common import imgutil, logutil

import matplotlib.pyplot as plt

import thutil
from thutil import test_shape, theano_function, maximum

from net import *
from pooling import cmrnorm, sum_unpool_2d
from utils import zero_pad_batch

from dataset_defs import NYUDepthModelDefs

_log = logutil.getLogger()
xx = np.newaxis

class machine(Machine, NYUDepthModelDefs):
    def __init__(self, conf):
        self.define_meta()
        Machine.__init__(self, conf)

    def infer_depth_and_normals(self, images):
        '''
        Infers depth and normals maps for a list of 320x240 images.
        images is a nimgs x 240 x 320 x 3 numpy uint8 array.
        returns depths and normals corresponding to the center box
        in the original rgb image.
        '''
        images = images.transpose((0,3,1,2))
        (nimgs, nc, nh, nw) = images.shape
        assert (nc, nh, nw) == (3,) + self.orig_input_size

        (input_h, input_w) = self.input_size
        (output_h, output_w) = self.output_size

        bsize = self.bsize
        b = 0

        # theano function for inference
        v = self.vars
        pred_depths = self.inverse_depth_transform(self.scale3.depths.pred_mean)
        pred_normals = self.scale3.normals.pred_mean
        infer_f = theano.function([v.images], (pred_depths, pred_normals))

        depths = np.zeros((nimgs, output_h, output_w), dtype=np.float32)
        normals = np.zeros((nimgs, 3, output_h, output_w), dtype=np.float32)

        # crop region (from random translations in training)
        dh = nh - input_h
        dw = nw - input_w
        (i0, i1) = (dh/2, nh - dh/2)
        (j0, j1) = (dw/2, nw - dw/2)

        # infer depth for images in batches
        b = 0
        while b < nimgs:
            batch = images[b:b+bsize]
            n = len(batch)
            if n < bsize:
                batch = zero_pad_batch(batch, bsize)

            # crop to network input size
            batch = batch[:, :, i0:i1, j0:j1]

            # infer depth with nnet
            (batch_depths, batch_normals) = infer_f(batch)
            depths[b:b+n] = batch_depths[:n, 0]
            normals[b:b+n] = batch_normals[:n]
            
            b += n

        return (depths, normals)

    def inverse_depth_transform(self, logdepths):
        # map network output log depths back to depth
        # output bias is init'd with the mean, and output is logdepth / stdev
        return T.exp(logdepths * self.meta.logdepths_std)

    def get_predicted_region(self):
        '''
        Returns the region of a 320x240 image covered by the predicted
        depth map (y0 y1 x0 x1) where y runs the 240-dim and x runs the 320-dim.
        '''
        # found using trace-back of pixel inds through train target
        return (11, 227, 13, 305)

    def make_test_values(self):
        (input_h, input_w) = self.input_size
        (output_h, output_w) = self.output_size
        test_images_size = (self.bsize, 3, input_h, input_w)
        test_depths_size = (self.bsize, output_h, output_w)
        test_normals_size = (self.bsize, 3, output_h, output_w)

        test_values = {}
        test_values['images'] = \
            (255 * np.random.rand(*test_images_size)).astype(np.float32)
        test_values['depths'] = \
            np.random.randn(*test_depths_size).astype(np.float32)
        test_values['normals'] = \
            np.random.randn(*test_normals_size).astype(np.float32)
        test_values['masks'] = \
            np.ones(test_depths_size, dtype=np.float32)
        return test_values

    def define_machine(self):
        self.scale2_size = self.conf.geteval('train', 'scale2_size')
        self.scale3_size = self.conf.geteval('train', 'scale3_size')
        (input_h, input_w) = self.input_size
        (scale2_h, scale2_w) = self.scale2_size
        (scale3_h, scale3_w) = self.scale3_size
        self.output_size = self.scale3_size

        # input vars
        images = T.tensor4('input')
        depths_target = T.tensor3('depths_target')
        normals_target = T.tensor4('normals_target')
        masks = T.tensor3('masks')

        test_values = self.make_test_values()
        images.tag.test_value = test_values['images']
        depths_target.tag.test_value = test_values['depths']
        normals_target.tag.test_value = test_values['normals']
        masks.tag.test_value  = test_values['masks']
       
        x0 = images
        depths0 = depths_target
        normals0 = normals_target
        m0 = masks

        self.inputs = MachinePart(locals())

        # downsample depth and mask by 2x
        m0 = m0[:,1::2,1::2][:,:-1,:-1]
        depths0 = depths0[:,1::2,1::2][:,:-1,:-1]
        normals0 = normals0[:,:,1::2,1::2][:,:,:-1,:-1]

        # features for scale1 stack from imagenet
        self.define_imagenet_stack(x0)

        # pretrained features are rather large, rescale down to nicer range
        imnet_r5 = 0.01 * self.imagenet.r5
        imnet_feats = imnet_r5.reshape((
                            self.bsize, T.prod(imnet_r5.shape[1:])))

        # rest of scale1 stack
        self.define_scale1_stack(imnet_feats)

        # scale2 stack
        self.define_scale2_stack(x0)

        # scale3 stack
        self.define_scale3_stack(x0)

        self.vars = MachinePart(locals())

    def define_imagenet_stack(self, x0):
        conv1 = self.create_unit('imnet_conv1')
        pool1 = self.create_unit('imnet_pool1')
        conv2 = self.create_unit('imnet_conv2')
        pool2 = self.create_unit('imnet_pool2')
        conv3 = self.create_unit('imnet_conv3')
        conv4 = self.create_unit('imnet_conv4')
        conv5 = self.create_unit('imnet_conv5')
        pool5 = self.create_unit('imnet_pool5')

        z1 = conv1.infer(x0 - 128)
        (p1, s1) = pool1.infer(z1)
        r1 = cmrnorm(relu(p1))

        z2 = conv2.infer(r1)
        (p2, s2) = pool2.infer(z2)
        r2 = cmrnorm(relu(p2))

        z3 = conv3.infer(r2)
        r3 = relu(z3)

        z4 = conv4.infer(r3)
        r4 = relu(z4)

        z5 = conv5.infer(r4)
        (p5, s5) = pool5.infer(z5)
        r5 = relu(p5)

        #r5_vec = r5.reshape((r5.shape[0], T.prod(r5.shape[1:])))
        #full6 = self.create_unit('imnet_full6',
        #                         ninput=test_shape(r5_vec)[1])
        #z6 = 0.5 * full6.infer(r5_vec)
        #r6 = relu(z6)

        #full7 = self.create_unit('imnet_full7', ninput=test_shape(r6)[1])
        #z7 = 0.5 * full7.infer(r6)
        #r7 = relu(z7)

        #full8 = self.create_unit('imnet_full8', ninput=test_shape(r7)[1])
        #z8 = full8.infer(r7)

        #output = softmax(z8, axis=1)

        self.imagenet = MachinePart(locals())

    def define_scale1_stack(self, imnet_feats):
        full1 = self.create_unit('full1',
                                 ninput=test_shape(imnet_feats)[1])
        f_1 = relu(full1.infer(imnet_feats))
        f_1_drop = random_zero(f_1, 0.5)
        f_1_mean = 0.5 * f_1

        (fh, fw) = self.scale2_size
        full2 = self.create_unit('full2',
                                 ninput=test_shape(f_1_mean)[1])
        
        f_2_drop = relu(full2.infer(f_1_drop))
        f_2_mean = relu(full2.infer(f_1_mean))

        (fh, fw) = self.scale2_size
        full2_feature_size = self.conf.geteval('full2', 'feature_size')
        (nfeat, nh, nw) = full2_feature_size
        assert (nh, nw) == (14, 19) and (fh, fw) == (55, 74)

        # upsample feature maps to scale2 size
        f_2_drop = f_2_drop.reshape((self.bsize, nfeat, nh, nw))
        f_2_mean = f_2_mean.reshape((self.bsize, nfeat, nh, nw))
        f_2_drop_up = upsample_bilinear(f_2_drop, 4)[:, :, 2:-2, 2:-3]
        f_2_mean_up = upsample_bilinear(f_2_mean, 4)[:, :, 2:-2, 2:-3]
        assert test_shape(f_2_drop_up)[-2:] == (fh, fw)

        self.scale1 = MachinePart(locals())

    def define_scale2_stack(self, x0):
        # input features
        x0_pproc = (x0 - self.meta.images_mean) \
                   * self.meta.images_istd

        conv_s2_1 = self.create_unit('conv_s2_1')
        z_s2_1    = relu(conv_s2_1.infer(x0_pproc))

        pool_s2_1 = self.create_unit('pool_s2_1')
        (p_s2_1, s_s2_1) = pool_s2_1.infer(z_s2_1)

        # concat input features with scale1 prediction
        p_1_drop = T.concatenate((self.scale1.f_2_drop_up,
                                  p_s2_1),
                                 axis=1)
        p_1_mean = T.concatenate((self.scale1.f_2_mean_up,
                                  p_s2_1),
                                 axis=1)

        # normals conv stack
        normals = self.define_scale2_onestack(
                            'normals', p_1_drop, p_1_mean)

        depths = self.define_scale2_onestack(
                            'depths', p_1_drop, p_1_mean)

        self.scale2 = MachinePart(locals())

    def define_scale2_onestack(self, stack_type, p_1_drop, p_1_mean):
        conv_s2_2 = self.create_unit('%s_conv_s2_2' % stack_type)
        z_s2_2_drop    = relu(conv_s2_2.infer(p_1_drop))
        z_s2_2_mean    = relu(conv_s2_2.infer(p_1_mean))

        conv_s2_3 = self.create_unit('%s_conv_s2_3' % stack_type)
        z_s2_3_drop    = relu(conv_s2_3.infer(z_s2_2_drop))
        z_s2_3_mean    = relu(conv_s2_3.infer(z_s2_2_mean))

        conv_s2_4 = self.create_unit('%s_conv_s2_4' % stack_type)
        z_s2_4_drop    = relu(conv_s2_4.infer(z_s2_3_drop))
        z_s2_4_mean    = relu(conv_s2_4.infer(z_s2_3_mean))

        conv_s2_5 = self.create_unit('%s_conv_s2_5' % stack_type)
        z_s2_5_drop    = conv_s2_5.infer(z_s2_4_drop)
        z_s2_5_mean    = conv_s2_5.infer(z_s2_4_mean)

        # prediction
        pred_drop = z_s2_5_drop
        pred_mean = z_s2_5_mean

        # add depths bias
        if stack_type == 'depths':
            depths_bias = self.create_unit('depths_bias', ninput=1)
            pred_drop += \
              depths_bias.bias.reshape(pred_drop.shape[1:], ndim=3)[xx,:,:,:]
            pred_mean += \
              depths_bias.bias.reshape(pred_mean.shape[1:], ndim=3)[xx,:,:,:]

        # unit normalize normals prediction vectors
        if stack_type == 'normals':
            pred_drop = (pred_drop / T.sqrt(T.sum(pred_drop**2, axis=1)
                                            + 1e-4)[:,xx,:,:])
            pred_mean = (pred_mean / T.sqrt(T.sum(pred_mean**2, axis=1)
                                            + 1e-4)[:,xx,:,:])

        return MachinePart(locals())

    def define_scale3_training_crop(self, output_size, crop_size):
        (oh, ow) = output_size
        (ch, cw) = crop_size
        rh = T.floor(theano_rng.uniform() * (oh - ch)).astype('int32')
        rw = T.floor(theano_rng.uniform() * (ow - cw)).astype('int32')
        rh.tag.test_value = np.int32(0.2 * (oh - ch))
        rw.tag.test_value = np.int32(0.2 * (ow - cw))
        x0 = self.inputs.x0
        x0_crop = x0[:,:,2*rh:2*(rh+ch+1)+8,2*rw:2*(rw+cw+1)+8]
        return (slice(rh, rh+ch), slice(rw, rw+cw), x0_crop)

    def define_scale3_stack(self, x0):
        # input features
        (fh, fw) = self.scale2_size
        (crop_h, crop_w, x0_crop) = \
                self.define_scale3_training_crop((2*fh-1, 2*fw-1),
                                                 (fh, fw))

        x0_pproc = (x0 - self.meta.images_mean) \
                    * self.meta.images_istd

        x0_pproc_crop = (x0_crop - self.meta.images_mean) \
                         * self.meta.images_istd

        conv_s3_1 = self.create_unit('conv_s3_1')
        pool_s3_1 = self.create_unit('pool_s3_1')

        z_s3_1    = relu(conv_s3_1.infer(x0_pproc))
        (p_s3_1, s_s3_1) = pool_s3_1.infer(z_s3_1)

        z_s3_1_crop = relu(conv_s3_1.infer(x0_pproc_crop))
        (p_s3_1_crop, s_s3_1_crop) = pool_s3_1.infer(z_s3_1_crop)

        def _upsamp_2_to_3(x):
            return thutil.constant(upsample_constant(x, 2)[:,:,:-1,:-1])

        # concat input features with scale2 prediction
        p_1_drop = T.concatenate(
                        (_upsamp_2_to_3(self.scale2.depths.pred_drop)
                            [:, :, crop_h, crop_w],
                         _upsamp_2_to_3(self.scale2.normals.pred_drop)
                            [:, :, crop_h, crop_w],
                         p_s3_1_crop[:, 4:, :, :]),
                        axis=1)
        p_1_mean = T.concatenate(
                        (_upsamp_2_to_3(self.scale2.depths.pred_mean),
                         _upsamp_2_to_3(self.scale2.normals.pred_mean),
                         p_s3_1[:, 4:, :, :]),
                        axis=1)

        # normals conv stack
        normals = self.define_scale3_onestack(
                            'normals', p_1_drop, p_1_mean,
                            crop_size=self.scale2_size)

        # depths conv stack
        depths = self.define_scale3_onestack(
                            'depths', p_1_drop, p_1_mean,
                            crop_size=self.scale2_size)

        self.scale3 = MachinePart(locals())

    def define_scale3_onestack(self, stack_type, p_2_drop, p_2_mean, crop_size):
        conv_s3_2 = self.create_unit('%s_conv_s3_2' % stack_type)
        conv_s3_3 = self.create_unit('%s_conv_s3_3' % stack_type)
        conv_s3_4 = self.create_unit('%s_conv_s3_4' % stack_type)

        z_s3_2_drop    = relu(conv_s3_2.infer(p_2_drop))
        z_s3_2_mean    = relu(conv_s3_2.infer(p_2_mean))

        z_s3_3_drop    = relu(conv_s3_3.infer(z_s3_2_drop))
        z_s3_3_mean    = relu(conv_s3_3.infer(z_s3_2_mean))

        z_s3_4_drop    = conv_s3_4.infer(z_s3_3_drop)
        z_s3_4_mean    = conv_s3_4.infer(z_s3_3_mean)

        # prediction
        pred_drop = z_s3_4_drop
        pred_mean = z_s3_4_mean

        # unit normalize normals prediction vectors
        if stack_type == 'normals':
            pred_drop = (pred_drop / T.sqrt(T.sum(pred_drop**2, axis=1)
                                            + 1e-4)[:,xx,:,:])
            pred_mean = (pred_mean / T.sqrt(T.sum(pred_mean**2, axis=1)
                                            + 1e-4)[:,xx,:,:])

        return MachinePart(locals())


    def define_depths_cost(self, pred, y0, m0):
        bsize = self.bsize
        npix = int(np.prod(test_shape(y0)[1:]))
        y0_target_vec = y0.reshape((self.bsize, npix))
        y0_mask_vec = m0.reshape((self.bsize, npix))
        pred_vec = pred.reshape((self.bsize, npix))

        # avg l2 + scale inv loss + spatial grad cost
        
        p = pred_vec * y0_mask_vec
        t = y0_target_vec * y0_mask_vec

        d = (p - t)

        nvalid_pix = T.sum(y0_mask_vec, axis=1)
        depth_error = (T.sum(nvalid_pix * T.sum(d**2, axis=1))
                         - 0.5*T.sum(T.sum(d, axis=1)**2)) \
                      / T.maximum(T.sum(nvalid_pix**2), 1)
        depth_cost = depth_error

        if pred.ndim == 4:
            pred = pred[:,0,:,:]
        if y0.ndim == 4:
            y0 = y0[:,0,:,:]
        if m0.ndim == 4:
            m0 = m0[:,0,:,:]

        h = 1
        p_di = (pred[:,h:,:] - pred[:,:-h,:]) * (1 / np.float32(h))
        p_dj = (pred[:,:,h:] - pred[:,:,:-h]) * (1 / np.float32(h))
        t_di = (y0[:,h:,:] - y0[:,:-h,:]) * (1 / np.float32(h))
        t_dj = (y0[:,:,h:] - y0[:,:,:-h]) * (1 / np.float32(h))
        m_di = T.logical_and(m0[:,h:,:], m0[:,:-h,:])
        m_dj = T.logical_and(m0[:,:,h:], m0[:,:,:-h])

        grad_cost = T.sum(m_di * (p_di - t_di)**2) / T.sum(m_di) \
                  + T.sum(m_dj * (p_dj - t_dj)**2) / T.sum(m_dj)

        depth_error += grad_cost
        depth_cost = depth_error

        return (depth_error, depth_cost)

