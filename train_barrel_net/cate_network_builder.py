from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tfutils import model

def getEncodeConvFilterSize(i, cfg):
    tmp_dict = cfg["alllayer"]["l%i" % i]["conv"]
    if "filter_size1" in tmp_dict:
        return (tmp_dict["filter_size1"], tmp_dict["filter_size2"])
    else:
        return tmp_dict["filter_size"]

def getEncodeConvNumFilters(i, cfg):
    tmp_dict = cfg["alllayer"]["l%i" % i]["conv"]
    return tmp_dict["num_filters"]

'''
def catenet(inputs, cfg_initial, train=True, seed = None, **kwargs):
    """The Model definition."""

    input_force, input_torque = inputs

    cfg = cfg_initial

    if seed==None:
        fseed = 0
    else:
        fseed = seed

    dropout_rate = 0.5
    if not train:
        dropout_rate = None

    print(input_force.get_shape().as_list())
    print(input_torque.get_shape().as_list())
    layer_depth = cfg['layernum']
    print('Layer depth: %d' % layer_depth)

    m = CateNetfromConv(seed = fseed, **kwargs)

    with tf.contrib.framework.arg_scope([m.conv], init='xavier',
                                        stddev=.01, bias=0, activation='relu'):
        for i in xrange(layer_depth):
            if 'conv' in cfg["alllayer"]["l%i" % i]:
                with tf.variable_scope('conv%i' % i):
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
                    encode_nodes.append(new_encode_node)   

        #hidden
        hidden_depth = getHiddenDepth(cfg)

        for i in range(1, hidden_depth + 1):
            with tf.variable_scope('hid%i' % (i + encode_depth)):
                nf = getHiddenNumFeatures(i, cfg)
                m.fc(nf, init='trunc_norm', dropout=dropout_rate, bias=.1)
                print('hidden layer %d %d' % (i, nf))

        #decode
        decode_depth = getDecodeDepth(cfg)
        print('Decode depth: %d' % decode_depth)

        nf = getDecodeNumFilters(0, decode_depth, cfg)
        ds = getDecodeSize(0, cfg)

        with tf.variable_scope('trans%i' % (encode_depth + encode_depth)):
            m.fc(ds*ds*nf, init='trunc_norm', dropout=None, activation=None, bias=.1)
            print("Linear to %d for input size %d" % (ds * ds * nf, ds))

        decode = m.reshape([ds, ds, nf])    
        print("Unflattening to", decode.get_shape().as_list())

        for i in range(1, decode_depth + 1):
            with tf.variable_scope('dec%i' % (encode_depth + encode_depth + i)):
                ds = getDecodeSize(i, cfg)

                if i == decode_depth:
                     assert ds == IMAGE_SIZE, (ds, IMAGE_SIZE)

                decode = m.resize_images(ds)
                
                print('Decode resize %d to shape' % i, decode.get_shape().as_list())

                add_bypass = getDecodeBypass(i, encode_nodes, ds, 1, cfg)

                if add_bypass != None:
                    bypass_layer = encode_nodes[add_bypass]

                    decode = m.add_bypass(bypass_layer)

                    print('Decode bypass from %d at %d for shape' % (add_bypass, i), decode.get_shape().as_list())

                cfs = getDecodeFilterSize(i, cfg)
                nf = getDecodeNumFilters(i, decode_depth, cfg)

                if i == decode_depth:
                    assert nf == NUM_CHANNELS, (nf, NUM_CHANNELS)

                if i < decode_depth:
                    decode = m.conv(nf, cfs, 1)
                else:
                    decode = m.conv(nf, cfs, 1, activation = None)

                print('Decode conv %d with size %d numfilters %d for shape' % (i, cfs, nf), decode.get_shape().as_list())

    return m
'''

def catenet(cfg_initial = None, train=True, seed=0, **kwargs):
    defaults = {'conv': {'batch_norm': False,
                         'kernel_init': 'xavier',
                         'kernel_init_kwargs': {'seed': seed}},
                         'weight_decay': .0005,
                'max_pool': {'padding': 'SAME'},
                'fc': {'batch_norm': False,
                       'kernel_init': 'truncated_normal',
                       'kernel_init_kwargs': {'stddev': .01, 'seed': seed},
                       'weight_decay': .0005,
                       'dropout_seed': 0}}
    m = model.ConvNet(defaults=defaults)
    dropout = .5 if train else None

    m.conv(96, (9, 3), 1, padding='VALID', layer='conv1')

    m.max_pool((3,1), (3,1), layer='conv1')

    m.conv(256, 3, 1, layer='conv2')
    m.max_pool(3, 2, layer='conv2')

    m.conv(384, 3, 1, layer='conv3')
    m.conv(384, 3, 1, layer='conv4')

    m.conv(256, 3, 1, layer='conv5')
    m.max_pool(3, 2, layer='conv5')

    m.fc(4096, dropout=dropout, bias=.1, layer='fc6')
    m.fc(1024, dropout=dropout, bias=.1, layer='fc7')

    m_add = model.ConvNet(defaults=defaults)
    m_add.fc(117, activation=None, dropout=None, bias=0, layer='fc8')

    return m, m_add

def catenet_add(cfg_initial = None, train=True, seed=0, **kwargs):
    defaults = {'conv': {'batch_norm': False,
                         'kernel_init': 'xavier',
                         'kernel_init_kwargs': {'seed': seed}},
                         'weight_decay': .0005,
                'max_pool': {'padding': 'SAME'},
                'fc': {'batch_norm': False,
                       'kernel_init': 'truncated_normal',
                       'kernel_init_kwargs': {'stddev': .01, 'seed': seed},
                       'weight_decay': .0005,
                       'dropout_seed': 0}}
    m_add = model.ConvNet(defaults=defaults)
    m_add.fc(117, activation=None, dropout=None, bias=0, layer='fc8')
    return m_add
'''
def catenet_tfutils(inputs, **kwargs):
    m = catenet((inputs['Data_force'], inputs['Data_torque']), **kwargs)
    return m.output, m.params
'''

def catenet_tfutils(inputs, **kwargs):

    # Deal with inputs
    input_force, input_torque = (inputs['Data_force'], inputs['Data_torque'])
    shape_list = input_force.get_shape().as_list()
    input_force_rs = tf.reshape(input_force, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_torque_rs = tf.reshape(input_torque, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_con = tf.concat([input_force_rs, input_torque_rs], 3)
    print(input_con.get_shape().as_list())
    input1,input2,input3 = tf.split(input_con, 3, 1)

    with tf.variable_scope("cate_root"):
        # Building the network

        with tf.variable_scope("create"):
            m1, m_add_tmp = catenet(**kwargs)
            output_1 = m1(input1)
            output_tmp = m_add_tmp(tf.concat([output_1, output_1, output_1], 1))

        #tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("create", reuse=True):

            m2, m_add = catenet(**kwargs)
            output_2 = m2(input2)

            m3, m_add = catenet(**kwargs)
            output_3 = m3(input3)

            output_t = m_add(tf.concat([output_1, output_2, output_3], 1))
            #output_t = m_add(m1(input1))

        return output_t, m1.params
