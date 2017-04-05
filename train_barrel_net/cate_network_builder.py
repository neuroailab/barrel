from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tfutils import model

def getWhetherConv(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    ret_val = "conv" in tmp_dict
    return ret_val

def getConvFilterSize(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    if "filter_size1" in tmp_dict:
        return (tmp_dict["filter_size1"], tmp_dict["filter_size2"])
    else:
        return tmp_dict["filter_size"]

def getConvStride(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    if "stride1" in tmp_dict:
        return (tmp_dict["stride1"], tmp_dict["stride2"])
    else:
        return tmp_dict["stride"]

def getConvNumFilters(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["conv"]
    return tmp_dict["num_filters"]

def getWhetherPool(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    ret_val = "pool" in tmp_dict
    return ret_val

def getPoolFilterSize(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["pool"]
    if "filter_size1" in tmp_dict:
        return (tmp_dict["filter_size1"], tmp_dict["filter_size2"])
    else:
        return tmp_dict["filter_size"]

def getPoolStride(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["pool"]
    if "stride1" in tmp_dict:
        return (tmp_dict["stride1"], tmp_dict["stride2"])
    else:
        return tmp_dict["stride"]

def getFcNumFilters(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]["fc"]
    return tmp_dict["num_features"]

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

    cfg = cfg_initial

    layernum_sub = cfg['layernum_sub']
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer)
            if indx_layer==0:
                m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg),
                       padding='VALID', layer= layer_name)
            else:
                m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg),
                       layer= layer_name)

            do_pool = getWhetherPool(indx_layer, cfg)
            if do_pool:
                m.max_pool(getPoolFilterSize(indx_layer, cfg), getPoolStride(indx_layer, cfg), layer = layer_name)
        else:
            layer_name = "fc%i" % (1 + indx_layer)
            m.fc(getFcNumFilters(indx_layer, cfg), dropout=dropout, bias=.1, layer=layer_name)

    m_add = model.ConvNet(defaults=defaults)
    m_add.fc(117, activation=None, dropout=None, bias=0, layer='fc8')

    return m, m_add

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
