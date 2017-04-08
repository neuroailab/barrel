from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#from tfutils import model
import model

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

def getWhetherBn(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    return 'bn' in tmp_dict

def getBnMode(i, cfg, key_want = "subnet"):
    tmp_dict = cfg[key_want]["l%i" % i]
    return tmp_dict['bn']

def catenet(inputs, input_flag, cfg_initial = None, train=True, **kwargs):
    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    layernum_sub = cfg['layernum_sub']
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer)
            with tf.variable_scope(layer_name):
                if indx_layer==0:
                    m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg),
                           padding='VALID', in_layer = inputs)
                else:
                    m.conv(getConvNumFilters(indx_layer, cfg), getConvFilterSize(indx_layer, cfg), getConvStride(indx_layer, cfg))

                if getWhetherBn(indx_layer, cfg):
                    m.batchnorm(train, getBnMode(indx_layer, cfg))

                do_pool = getWhetherPool(indx_layer, cfg)
                if do_pool:
                    m.pool(getPoolFilterSize(indx_layer, cfg), getPoolStride(indx_layer, cfg))

        else:
            layer_name = "fc%i" % (1 + indx_layer)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg), init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg):
                    m.batchnorm(train, getBnMode(indx_layer, cfg))

    return m

def catenet_add(inputs, cfg_initial = None, train=True, **kwargs):

    m_add = model.ConvNet(**kwargs)
    with tf.variable_scope('fc_add'):
        m_add.fc(117, init='trunc_norm', activation=None, dropout=None, bias=0, in_layer=inputs)

    return m_add

def catenet_tfutils(inputs, **kwargs):

    # Deal with inputs
    input_force, input_torque = (inputs['Data_force'], inputs['Data_torque'])
    shape_list = input_force.get_shape().as_list()
    input_force_rs = tf.reshape(input_force, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_torque_rs = tf.reshape(input_torque, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_con = tf.concat([input_force_rs, input_torque_rs], 3)
    print(input_con.get_shape().as_list())
    input1,input2,input3 = tf.split(input_con, 3, 1)

    input_flag = tf.equal(inputs['trainflag'][0], tf.constant(1, dtype=tf.int64))

    with tf.variable_scope("cate_root"):
        # Building the network

        with tf.variable_scope("create"):
            m1 = catenet(input1, input_flag, **kwargs)
            output_1 = m1.output

        #tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("create", reuse=True):

            m2 = catenet(input2, input_flag, **kwargs)
            output_2 = m2.output

            m3 = catenet(input3, input_flag, **kwargs)
            output_3 = m3.output

        with tf.variable_scope("create"):
            input_t = tf.concat([output_1, output_2, output_3], 1)
            m_final = catenet_add(input_t, **kwargs)

        return m_final.output, m_final.params
