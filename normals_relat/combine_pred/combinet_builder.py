from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys

sys.path.append('../normal_pred/')
from normal_encoder_asymmetric_with_bypass import *

def getWhetherBn(i, cfg, key_want = "encode"):
    tmp_dict = cfg[key_want][i]
    return 'bn' in tmp_dict

def normal_vgg16_forcombine(inputs, cfg_initial, train=True, seed = None, center_im = False, reuse_flag = None, reuse_batch = None, batch_name = '', **kwargs):
    """The Model definition for normals"""

    cfg = cfg_initial
    if seed==None:
        fseed = getFilterSeed(cfg)
    else:
        fseed = seed

    if center_im:
        inputs  = tf.subtract(inputs, tf.constant(0.5, dtype=tf.float32))

    m = NoramlNetfromConv(seed = fseed, **kwargs)

    encode_nodes = []
    encode_nodes.append(inputs)

    with tf.contrib.framework.arg_scope([m.conv], init='xavier',
                                        stddev=.01, bias=0, activation='relu'):
        encode_depth = getEncodeDepth(cfg)
        print('Encode depth: %d' % encode_depth)

        for i in range(1, encode_depth + 1):
            with tf.variable_scope('encode%i' % i, reuse=reuse_flag):
                cfs = getEncodeConvFilterSize(i, cfg)
                nf = getEncodeConvNumFilters(i, cfg)
                cs = getEncodeConvStride(i, encode_depth, cfg)

                if i==1:
                    new_encode_node = m.conv(nf, cfs, cs, padding='VALID', in_layer=inputs)
                else:
                    new_encode_node = m.conv(nf, cfs, cs)

                print('Encode conv %d with size %d stride %d numfilters %d' % (i, cfs, cs, nf))        
                do_pool = getEncodeDoPool(i, cfg)
                if do_pool:
                    pfs = getEncodePoolFilterSize(i, cfg)
                    ps = getEncodePoolStride(i, cfg)
                    pool_type = getEncodePoolType(i, cfg)

                    if pool_type == 'max':
                        pfunc = 'maxpool'
                    elif pool_type == 'avg':
                        pfunc = 'avgpool' 

                    new_encode_node = m.pool(pfs, ps, pfunc=pfunc)
                    print('Encode %s pool %d with size %d stride %d' % (pfunc, i, pfs, ps))
            if getWhetherBn(i, cfg):
                with tf.variable_scope('encode_bn%i%s' % (i, batch_name), reuse=reuse_batch):
                    new_encode_node = m.batchnorm_corr(train)

            encode_nodes.append(new_encode_node)   

        decode_depth = getDecodeDepth(cfg)
        print('Decode depth: %d' % decode_depth)

        for i in range(1, decode_depth + 1):
            with tf.variable_scope('decode%i' % (encode_depth + i), reuse=reuse_flag):

                add_bypass = getDecodeBypass(i, encode_nodes, None, 0, cfg)

                if add_bypass != None:
                    bypass_layer = encode_nodes[add_bypass]

                    decode = m.add_bypass(bypass_layer)

                    print('Decode bypass from %d at %d for shape' % (add_bypass, i), decode.get_shape().as_list())

                do_unpool = getDecodeDoUnPool(i, cfg)
                if do_unpool:
                    unpool_scale = getDecodeUnPoolScale(i, cfg)
                    new_encode_node = m.resize_images_scale(unpool_scale)

                    print('Decode unpool %d with scale %d' % (i, unpool_scale))

                cfs = getEncodeConvFilterSize(i, cfg, which_one = 'decode')
                nf = getEncodeConvNumFilters(i, cfg, which_one = 'decode')
                cs = getEncodeConvStride(i, encode_depth, cfg, which_one = 'decode')

                new_encode_node = m.conv(nf, cfs, cs)

                print('Decode conv %d with size %d stride %d numfilters %d' % (i, cfs, cs, nf))        

    return m

def combine_normal_tfutils(inputs, **kwargs):
    #print(inputs.keys())
    #print(inputs['image_t'].get_shape().as_list(), inputs['image_t'].dtype)
    inputs_t = tf.cast(inputs['image_t'], tf.float32)
    m_t = normal_vgg16_forcombine(inputs_t, reuse_flag = None, reuse_batch = None, batch_name = '_t', **kwargs)

    inputs_s = tf.cast(inputs['image_s'], tf.float32)

    m_s = normal_vgg16_forcombine(inputs_s, reuse_flag = True, reuse_batch = None, batch_name = '_s', **kwargs)

    return [m_t.output, m_s.output], m_s.params
