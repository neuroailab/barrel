from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os, sys

#from tfutils import model
sys.path.append('../../train_barrel_net/')
import model

IMAGE_SIZE = 224
NUM_CHANNELS = 3

class NoramlNetfromConv(model.ConvNet):
    def __init__(self, seed=None, **kwargs):
        super(NoramlNetfromConv, self).__init__(seed=seed, **kwargs)

    @tf.contrib.framework.add_arg_scope
    def pool(self,
             ksize=3,
             stride=2,
             padding='SAME',
             in_layer=None,
             pfunc='maxpool'):
        if in_layer is None:
            in_layer = self.output

        if isinstance(ksize, int):
            ksize1 = ksize
            ksize2 = ksize
        else:
            ksize1, ksize2 = ksize

        if pfunc=='maxpool':
            self.output = tf.nn.max_pool(in_layer,
                                         ksize=[1, ksize1, ksize2, 1],
                                         strides=[1, stride, stride, 1],
                                         padding=padding,
                                         name='pool')
        else:
            self.output = tf.nn.avg_pool(in_layer,
                                         ksize=[1, ksize1, ksize2, 1],
                                         strides=[1, stride, stride, 1],
                                         padding=padding,
                                         name='pool')
        self.params = {'input': in_layer.name,
                       'type': pfunc,
                       'kernel_size': (ksize1, ksize2),
                       'stride': stride,
                       'padding': padding}
        return self.output

    def reshape(self, new_size, in_layer=None):
        if in_layer is None:
            in_layer = self.output

        size_l = [in_layer.get_shape().as_list()[0]]
        size_l.extend(new_size)
        self.output = tf.reshape(in_layer, size_l)
        return self.output

    def resize_images(self, new_size, in_layer=None):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.image.resize_images(in_layer, [new_size, new_size])
        return self.output

    def add_bypass(self, bypass_layer, in_layer=None):
        if in_layer is None:
            in_layer = self.output

        bypass_shape = bypass_layer.get_shape().as_list()
        ds = in_layer.get_shape().as_list()[1]
        if bypass_shape[1] != ds:
            bypass_layer = tf.image.resize_images(bypass_layer, [ds, ds])
        self.output = tf.concat([in_layer, bypass_layer], 3)

        return self.output

    def resize_images_scale(self, scale, in_layer=None):
        if in_layer is None:
            in_layer = self.output

        im_h = in_layer.get_shape().as_list()[1]
        im_w = in_layer.get_shape().as_list()[2]

        self.output = tf.image.resize_images(in_layer, [im_h*scale, im_w*scale])
        return self.output
    

def getBatchSize(cfg):
    val = 128
    if 'BATCH_SIZE' in cfg:
        val = cfg['BATCH_SIZE']
    return val

def getQueueCap(cfg):
    val = 5120
    if 'QUEUE_CAP' in cfg:
        val = cfg['QUEUE_CAP']
    return val

def getEncodeDepth(cfg):
    val = None
    if 'encode_depth' in cfg:
        val = cfg['encode_depth']
    elif 'encode' in cfg:
        val = max(cfg['encode'].keys())

    if val is not None:
        return val
    val = 2
    return val

def getEncodeConvFilterSize(i, cfg, prev=None, which_one = 'encode'):
    val = None

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'filter_size' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['filter_size']    

    if val is not None:
        return val

    val = 5
    return val

def getEncodeConvNumFilters(i, cfg, which_one = 'encode'):
    val = None

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'num_filters' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['num_filters']

    if val is not None:
        return val

    L = [3, 48, 96, 128, 256, 128]
    return L[i]

def getEncodeConvStride(i, encode_depth, cfg, which_one = 'encode'):
    val = None

    if which_one in cfg and (i in cfg[which_one]):
        if 'conv' in cfg[which_one][i]:
            if 'stride' in cfg[which_one][i]['conv']:
                val = cfg[which_one][i]['conv']['stride']

    if val is not None:
        return val

    if encode_depth > 1:
        return 2 if i == 1 else 1
    else:
        return 3 if i == 1 else 1

def getDecodeDoUnPool(i, cfg):
    val = None
    if 'decode' in cfg and (i in cfg['decode']):
        if 'unpool' in cfg['decode'][i]:
            val = True

    if val is not None:
        return val
    return False

def getDecodeUnPoolScale(i, cfg):
    val = None
    if 'decode' in cfg and (i in cfg['decode']):
        if 'unpool' in cfg['decode'][i]:
            if 'scale' in cfg['decode'][i]['unpool']:
                val = cfg['decode'][i]['unpool']['scale']

    if val is not None:
        return val

    return 2

def getEncodeDoPool(i, cfg):
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'do_pool' in cfg['encode'][i]:
            val = cfg['encode'][i]['do_pool']
        elif 'pool' in cfg['encode'][i]:
            val = True
    if val is not None:
        return val
    return False
    #return 1

def getEncodePoolFilterSize(i, cfg):
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'pool' in cfg['encode'][i]:
            if 'filter_size' in cfg['encode'][i]['pool']:
                val = cfg['encode'][i]['pool']['filter_size']

    if val is not None:
        return val
    L = [2, 2, 3, 3, 3]
    return L[i]

def getEncodePoolStride(i, cfg):    
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'pool' in cfg['encode'][i]:
            if 'stride' in cfg['encode'][i]['pool']:
                val = cfg['encode'][i]['pool']['stride']
    if val is not None:
        return val
    return 2

def getEncodePoolType(i, cfg):
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'pool' in cfg['encode'][i]:
            if 'type' in cfg['encode'][i]['pool']:
                val = cfg['encode'][i]['pool']['type']
    if val is not None:
        return val
    L = ['max', 'max', 'avg', 'avg', 'avg']
    if i<len(L):
        return L[i]
    else:
        return 'max'

def getHiddenDepth(cfg):
    val = None
    if 'hidden_depth' in cfg:
        val = cfg['hidden_depth']
    elif 'hidden' in cfg:
        val = max(cfg['hidden'].keys())

    if val is not None:
        return val
    return 2

def getHiddenNumFeatures(i, cfg):
    val = None
    if 'hidden' in cfg and (i in cfg['hidden']):
        if 'num_features' in cfg['hidden'][i]:
            val = cfg['hidden'][i]['num_features']
    if val is not None:
        return val
    return 1024

def getDecodeDepth(cfg):
    val = None
    if 'decode_depth' in cfg:
        val = cfg['decode_depth']
    elif 'decode' in cfg:
        val = max(cfg['decode'].keys())

    if val is not None:
        return val
    return 3

def getDecodeNumFilters(i, decode_depth,cfg):
    if i < decode_depth:
        val = None
        if 'decode' in cfg and (i in cfg['decode']):
            if 'num_filters' in cfg['decode'][i]:
                val = cfg['decode'][i]['num_filters']

        if val is not None:
            return val
        return 32
    else:
        return NUM_CHANNELS

def getDecodeFilterSize(i, cfg):
    val = None
    if 'decode' in cfg and (i in cfg['decode']):
         if 'filter_size' in cfg['decode'][i]:
             val = cfg['decode'][i]['filter_size']
             
    if val is not None:
        return val
    L = [1, 5, 1, 5, 5]
    return L[i]

def getDecodeSize(i, cfg):
    val = None
    if 'decode' in cfg and (i in cfg['decode']):
        if 'size' in cfg['decode'][i]:
            val = cfg['decode'][i]['size']
    if val is not None:
        return val
    l = [28, 56, 112, 224]
    return l[i]

def getDecodeBypass(i, encode_nodes, decode_size, switch, cfg):
    val = None
    if 'decode' in cfg and (i in cfg['decode']):
        if 'bypass' in cfg['decode'][i]:
            val = cfg['decode'][i]['bypass']

    if val is not None:
        return val 
    if switch==1:
        sdiffs = [e.get_shape().as_list()[1] - decode_size for e in encode_nodes]
        return np.abs(sdiffs).argmin()
    else:
        return None

def getFilterSeed(cfg):
    if 'filter_seed' in cfg:
        return cfg['filter_seed']
    else:    
        return 0
    
def normal_vgg16(inputs, cfg_initial, train=True, seed = None, center_im = False, **kwargs):
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
            with tf.variable_scope('encode%i' % i):
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

        decode_depth = getDecodeDepth(cfg)
        print('Decode depth: %d' % decode_depth)

        for i in range(1, decode_depth + 1):
            with tf.variable_scope('decode%i' % (encode_depth + i)):

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

def normal_vgg16_tfutils(inputs, **kwargs):
    m = normal_vgg16(inputs['images'], **kwargs)
    return m.output, m.params

def normalnet(inputs, cfg_initial, train=True, seed = None, **kwargs):
    """The Model definition."""

    cfg = cfg_initial
    if seed==None:
        fseed = getFilterSeed(cfg)
    else:
        fseed = seed

    dropout_rate = 0.5
    if not train:
        dropout_rate = None
        
    
    #encoding
    imsize = IMAGE_SIZE

    encode_depth = getEncodeDepth(cfg)
    print('Encode depth: %d' % encode_depth)

    m = NoramlNetfromConv(seed = fseed, **kwargs)

    encode_nodes = []
    encode_nodes.append(inputs)

    with tf.contrib.framework.arg_scope([m.conv], init='xavier',
                                        stddev=.01, bias=0, activation='relu'):
        for i in range(1, encode_depth + 1):
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

    #return m.output, m.params
    #return m.output, cfg
    return m

def normalnet_tfutils(inputs, **kwargs):
    m = normalnet(inputs['images'], **kwargs)
    return m.output, m.params


def get_model(rng, batch_size, cfg, slippage, slippage_error, host, port, datapath):
    global sock
    if sock is None:
        initialize(host, port, datapath)

    image_node = tf.placeholder(tf.float32,
                                                            shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    normals_node = tf.placeholder(tf.float32,
                                                                shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    
    train_prediction, cfg = model(image_node, rng, cfg, slippage=slippage, slippage_error=slippage_error)

    norm = (IMAGE_SIZE**2) * NUM_CHANNELS * batch_size
    loss = tf.nn.l2_loss(train_prediction - normals_node) / norm

    innodedict = {'images': image_node,
                                'normals': normals_node}

    outnodedict = {'train_predictions': train_prediction,
                                 'loss': loss}

    return outnodedict, innodedict, cfg
