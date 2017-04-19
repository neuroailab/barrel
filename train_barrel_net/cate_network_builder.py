from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tfutils import model as model_tfutils
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

def build_partnet_3d(m, cfg, key_layernum, key_subnet, inputs=None, layer_offset=0, dropout=None):
    layernum_sub = cfg[ key_layernum ]
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg, key_want = key_subnet)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                curr_size = getConvFilterSize(indx_layer, cfg, key_want = key_subnet) 
                curr_size_list = [1, curr_size, curr_size]
                if indx_layer==0 and (not inputs==None):
                    m.conv3(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            curr_size_list,
                            getConvStride(indx_layer, cfg, key_want = key_subnet),
                           padding='VALID', in_layer = inputs)
                else:
                    m.conv3(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            curr_size_list, 
                            getConvStride(indx_layer, cfg, key_want = key_subnet))

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

                do_pool = getWhetherPool(indx_layer, cfg, key_want = key_subnet)
                if do_pool:
                    m.pool(getPoolFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getPoolStride(indx_layer, cfg, key_want = key_subnet))

        else:
            layer_name = "fc%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg, key_want = key_subnet), 
                        init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

    return m

def build_partnet(m, cfg, key_layernum, key_subnet, inputs=None, layer_offset=0, dropout=None):
    layernum_sub = cfg[ key_layernum ]
    for indx_layer in xrange(layernum_sub):
        do_conv = getWhetherConv(indx_layer, cfg, key_want = key_subnet)
        if do_conv:
            layer_name = "conv%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                if indx_layer==0 and (not inputs==None):
                    m.conv(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            getConvFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getConvStride(indx_layer, cfg, key_want = key_subnet),
                           padding='VALID', in_layer = inputs)
                else:
                    m.conv(getConvNumFilters(indx_layer, cfg, key_want = key_subnet), 
                            getConvFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getConvStride(indx_layer, cfg, key_want = key_subnet))

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

                do_pool = getWhetherPool(indx_layer, cfg, key_want = key_subnet)
                if do_pool:
                    m.pool(getPoolFilterSize(indx_layer, cfg, key_want = key_subnet), 
                            getPoolStride(indx_layer, cfg, key_want = key_subnet))

        else:
            layer_name = "fc%i" % (1 + indx_layer + layer_offset)
            with tf.variable_scope(layer_name):
                m.fc(getFcNumFilters(indx_layer, cfg, key_want = key_subnet), 
                        init='trunc_norm', dropout=dropout, bias=.1)

                if getWhetherBn(indx_layer, cfg, key_want = key_subnet):
                    m.batchnorm_corr(train)

    return m

def catenet_spa_temp_3d(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    shape_list = inputs.get_shape().as_list()
    curr_layer = 0

    assert shape_list[2]==35, 'Must set expand==1'

    inputs = tf.reshape(inputs, [shape_list[0], shape_list[1], 5, 7, -1])
    m = build_partnet_3d(m, cfg, "layernum_spa", "spanet", inputs = inputs, layer_offset = curr_layer, dropout = dropout)
    new_input = m.output
    shape_list_tmp = new_input.get_shape().as_list()
    new_input = tf.reshape(new_input, [shape_list_tmp[0], shape_list_tmp[1], 1, -1])

    m.output = new_input
    curr_layer = curr_layer + cfg["layernum_spa"]
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet_spa_temp(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    shape_list = inputs.get_shape().as_list()
    small_inputs = tf.split(inputs, shape_list[1], 1)
    small_outputs = []
    curr_layer = 0

    assert shape_list[2]==35, 'Must set expand==1'

    first_flag = True

    for small_input in small_inputs:
        small_input = tf.reshape(small_input, [shape_list[0], 5, 7, -1])
        if first_flag:
            with tf.variable_scope("small"):
                m = build_partnet(m, cfg, "layernum_spa", "spanet", inputs = small_input, layer_offset = curr_layer, dropout = dropout)
            first_flag = False
        else:
            with tf.variable_scope("small", reuse=True):
                m = build_partnet(m, cfg, "layernum_spa", "spanet", inputs = small_input, layer_offset = curr_layer, dropout = dropout)
        small_output = m.output
        shape_list_tmp = small_output.get_shape().as_list()
        small_output = tf.reshape(small_output, [shape_list_tmp[0], 1, 1, -1])
        small_outputs.append(small_output)

    new_input = tf.concat(small_outputs, 1)
    m.output = new_input
    curr_layer = curr_layer + cfg["layernum_spa"]
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet_temp_spa(inputs, cfg_initial, train = True, **kwargs):

    m = model.ConvNet(**kwargs)

    cfg = cfg_initial

    dropout_default = 0.5
    if 'dropout' in cfg:
        dropout_default = cfg['dropout']

    dropout = dropout_default if train else None

    if dropout==0:
        dropout = None

    curr_layer = 0
    m = build_partnet(m, cfg, "layernum_temp", "tempnet", inputs = inputs, layer_offset = curr_layer, dropout = dropout)
    curr_layer = curr_layer + cfg["layernum_temp"]

    tensor_tmp = m.output
    tensor_tmp = tf.transpose(tensor_tmp, perm = [0, 2, 1, 3])

    shape_list = tensor_tmp.get_shape().as_list()
    tensor_tmp = tf.reshape(tensor_tmp, [shape_list[0], shape_list[1], -1])

    shape_now = tensor_tmp.get_shape().as_list()
    slice0 = tf.slice(tensor_tmp, [0, 0, 0], [-1, 5, -1])
    slice1 = tf.slice(tensor_tmp, [0, 5, 0], [-1, 6, -1])
    slice2 = tf.slice(tensor_tmp, [0, 11, 0], [-1, 14, -1])
    slice3 = tf.slice(tensor_tmp, [0, 25, 0], [-1, 6, -1])

    pad_ten0 = tf.zeros([shape_now[0], 1, shape_now[2]])
    pad_ten1 = tf.zeros([shape_now[0], 1, shape_now[2]])
    pad_ten2 = tf.zeros([shape_now[0], 1, shape_now[2]])
    pad_ten3 = tf.zeros([shape_now[0], 1, shape_now[2]])

    tensor_tmp = tf.concat([slice0, pad_ten0, pad_ten1, slice1, pad_ten2, slice2, pad_ten3, slice3], 1)

    tensor_tmp = tf.reshape(tensor_tmp, [shape_list[0], 5, 7, -1])

    m.output = tensor_tmp
    m = build_partnet(m, cfg, "layernum_spa", "spanet", layer_offset = curr_layer, dropout = dropout)

    return m

def catenet(inputs, cfg_initial = None, train=True, **kwargs):
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

def catenet_from_3s(input_con, func_each = catenet, **kwargs):
    input1,input2,input3 = tf.split(input_con, 3, 1)

    with tf.variable_scope("cate_root"):
        # Building the network

        with tf.variable_scope("create"):
            m1 = func_each(input1, **kwargs)
            output_1 = m1.output

        #tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("create", reuse=True):

            m2 = func_each(input2, **kwargs)
            output_2 = m2.output

            m3 = func_each(input3, **kwargs)
            output_3 = m3.output

        input_t = tf.concat([output_1, output_2, output_3], 1)

        return input_t

def catenet_from_12s(input_con, func_each = catenet, **kwargs):
    input0,input1,input2,input3 = tf.split(input_con, 4, 1)

    with tf.variable_scope("create_big"):
        input_t0 = catenet_from_3s(input0, func_each = func_each, **kwargs)

    with tf.variable_scope("create_big", reuse=True):
        input_t1 = catenet_from_3s(input1, func_each = func_each, **kwargs)
        input_t2 = catenet_from_3s(input2, func_each = func_each, **kwargs)
        input_t3 = catenet_from_3s(input3, func_each = func_each, **kwargs)

    input_t = tf.concat([input_t0, input_t1, input_t2, input_t3], 1)

    return input_t

def deal_with_inputs(inputs):
    # Deal with inputs
    input_force, input_torque = (inputs['Data_force'], inputs['Data_torque'])
    shape_list = input_force.get_shape().as_list()
    input_force_rs = tf.reshape(input_force, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_torque_rs = tf.reshape(input_torque, [shape_list[0], shape_list[1], shape_list[2], -1])
    input_con = tf.concat([input_force_rs, input_torque_rs], 3)

    return input_con


def catenet_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_temp_spa_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_temp_spa, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_temp_spa, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_spa_temp_3d_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_spa_temp_3d, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_spa_temp_3d, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

def catenet_spa_temp_tfutils(inputs, split_12 = False, **kwargs):

    input_con = deal_with_inputs(inputs)
    #print(input_con.get_shape().as_list())

    if not split_12:
        input_t = catenet_from_3s(input_con, func_each = catenet_spa_temp, **kwargs)
    else:
        input_t = catenet_from_12s(input_con, func_each = catenet_spa_temp, **kwargs)

    m_final = catenet_add(input_t, **kwargs)
    return m_final.output, m_final.params

# Deprecated
def catenet_old(cfg_initial = None, train=True, seed=0, **kwargs):
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
    m = model_tfutils.ConvNet(defaults=defaults)
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

    m_add = model_tfutils.ConvNet(defaults=defaults)
    m_add.fc(117, activation=None, dropout=None, bias=0, layer='fc8')

    return m, m_add

def catenet_tfutils_old(inputs, **kwargs):

    # Deal with inputs
    input_con = deal_with_inputs(inputs)
    print(input_con.get_shape().as_list())
    input1,input2,input3 = tf.split(input_con, 3, 1)

    with tf.variable_scope("cate_root"):
        # Building the network

        with tf.variable_scope("create"):
            m1 = catenet(input1, **kwargs)
            output_1 = m1.output

        #tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("create", reuse=True):

            m2 = catenet(input2, **kwargs)
            output_2 = m2.output

            m3 = catenet(input3, **kwargs)
            output_3 = m3.output

        with tf.variable_scope("create"):
            input_t = tf.concat([output_1, output_2, output_3], 1)
            m_final = catenet_add(input_t, **kwargs)

        return m_final.output, m_final.params
