from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tfutils import model

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

        size_l = [self.output.get_shape().as_list()[0]].extend(new_size)
        self.output = tf.reshape(in_layer, size_l)
        return self.output

    def resize_images(self, new_size, in_layer=None):
        if in_layer is None:
            in_layer = self.output
        self.output = tf.image.resize_images(in_layer, [new_size, new_size])
        return self.output

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

def getEncodeConvFilterSize(i, cfg, prev=None):
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'conv' in cfg['encode'][i]:
            if 'filter_size' in cfg['encode'][i]['conv']:
                val = cfg['encode'][i]['conv']['filter_size']    

    if val is not None:
        return val

    val = 5
    return val

def getEncodeConvNumFilters(i, cfg):
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'conv' in cfg['encode'][i]:
            if 'num_filters' in cfg['encode'][i]['conv']:
                val = cfg['encode'][i]['conv']['num_filters']

    if val is not None:
        return val

    L = [3, 48, 96, 128, 256, 128]
    return L[i]

def getEncodeConvStride(i, encode_depth, cfg):
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'conv' in cfg['encode'][i]:
            if 'stride' in cfg['encode'][i]['conv']:
                val = cfg['encode'][i]['conv']['stride']

    if val is not None:
        return val

    if encode_depth > 1:
        return 2 if i == 1 else 1
    else:
        return 3 if i == 1 else 1

def getEncodeDoPool(i, cfg):
    val = None
    if 'encode' in cfg and (i in cfg['encode']):
        if 'do_pool' in cfg['encode'][i]:
            val = cfg['encode'][i]['do_pool']
        elif 'pool' in cfg['encode'][i]:
            val = True
    if val is not None:
        return val
    return 1 

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
    return L[i]

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
    

def normalnet(inputs, train=True, **kwargs):
    """The Model definition."""

    fseed = getFilterSeed(cfg)
    
    #encoding
    nf0 = NUM_CHANNELS 
    imsize = IMAGE_SIZE

    encode_depth = getEncodeDepth(cfg)
    print('Encode depth: %d' % encode_depth)

    m = NoramlNetfromConv(seed = fseed, **kwargs)

    with tf.contrib.framework.arg_scope([m.conv], init='xavier',
                                        stddev=.01, bias=0, activation='relu'):
        for i in range(1, encode_depth + 1):
            with tf.variable_scope('conv%i' % i):
                cfs = getEncodeConvFilterSize(i, cfg)
                nf = getEncodeConvNumFilters(i, cfg)
                cs = getEncodeConvStride(i, encode_depth, cfg)

                if i==1:
                    m.conv(nf, cfs, cs, padding='VALID', in_layer=inputs)
                else:
                    m.conv(nf, cfs, cs)

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

                    m.pool(pfs, ps, pfunc)
                    print('Encode %s pool %d with size %d stride %d' % (pfunc, i, pfs, ps))

    #hidden
    hidden_depth = getHiddenDepth(cfg)

    for i in range(1, hidden_depth + 1):
        with tf.variable_scope('hid%i' i + encode_depth):
            nf = getHiddenNumFeatures(i, cfg)
            m.fc(nf, init='trunc_norm', dropout=.5, bias=.1)
            print('hidden layer %d %d' % (i, nf))

    #decode
    decode_depth = getDecodeDepth(cfg)
    print('Decode depth: %d' % decode_depth)

    nf = getDecodeNumFilters(0, decode_depth, cfg)
    ds = getDecodeSize(0, decode_depth, cfg)

    with tf.variable_scope('trans%i' encode_depth + encode_depth):
        m.fc(ds*ds*nf, init='trunc_norm', dropout=None, activation=None, bias=.1)
        print("Linear to %d for input size %d" % (ds * ds * nf, ds))

    decode = m.reshape([ds, ds, nf])    
    print("Unflattening to", decode.get_shape().as_list())

    for i in range(1, decode_depth + 1):
        nf0 = nf
        ds = getDecodeSize(i, cfg)

        if i == decode_depth:
             assert ds == IMAGE_SIZE, (ds, IMAGE_SIZE)
        decode = tf.image.resize_images(decode, ds, ds)
        
        print('Decode resize %d to shape' % i, decode.get_shape().as_list())
        add_bypass = getDecodeBypass(i, encode_nodes, ds, decode_depth, rng, cfg, slippage=slippage)
        if add_bypass != None:
            bypass_layer = encode_nodes[add_bypass]
            bypass_shape = bypass_layer.get_shape().as_list()
            if bypass_shape[1] != ds:
                bypass_layer = tf.image.resize_images(bypass_layer, ds, ds)
            decode = tf.concat(3, [decode, bypass_layer])
            print('Decode bypass from %d at %d for shape' % (add_bypass, i), decode.get_shape().as_list())
            nf0 = nf0 + bypass_shape[-1]
            cfg0['decode'][i]['bypass'] = add_bypass
        cfs = getDecodeFilterSize(i, decode_depth, rng, cfg, slippage=slippage)
        cfg0['decode'][i]['filter_size'] = cfs
        nf = getDecodeNumFilters(i, decode_depth, rng, cfg, slippage=slippage)
        cfg0['decode'][i]['num_filters'] = nf
        if i == decode_depth:
            assert nf == NUM_CHANNELS, (nf, NUM_CHANNELS)
        W = tf.Variable(tf.truncated_normal([cfs, cfs, nf0, nf],
                                                                                stddev=0.1,
                                                                                seed=fseed))
        b = tf.Variable(tf.zeros([nf]))
        decode = tf.nn.conv2d(decode,
                                                    W,
                                                    strides=[1, 1, 1, 1],
                                                    padding='SAME')
        decode = tf.nn.bias_add(decode, b)
        print('Decode conv %d with size %d num channels %d numfilters %d for shape' % (i, cfs, nf0, nf), decode.get_shape().as_list())

        if i < decode_depth:    #add relu to all but last ... need this?
            decode = tf.nn.relu(decode)

    return decode, cfg0


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