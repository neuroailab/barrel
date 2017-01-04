"""
Script used for training the normalnet using tfutils
"""

from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

import normal_encoder_asymmetric_with_bypass

from tfutils import base, data, model, optimizer

import json
import copy


host = os.uname()[1]
if host == 'freud':  # freud
    DATA_PATH = '/media/data/one_world_dataset/randomperm.hdf5'
else:
    print("Not supported yet!")
    exit()



def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k:[] for k in res}
    for k,v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def exponential_decay(global_step,
                      learning_rate=.01,
                      decay_factor=.95,
                      decay_steps=1,
                      ):
    # Decay the learning rate exponentially based on the number of steps.
    if decay_factor is None:
        lr = learning_rate  # just a constant.
    else:
        # Calculate the learning rate schedule.
        lr = tf.train.exponential_decay(
            learning_rate,  # Base learning rate.
            global_step,  # Current index into the dataset.
            decay_steps,  # Decay step
            decay_factor,  # Decay rate.
            staircase=True)
    return lr


class Threedworld(data.HDF5DataProvider):

    N_TRAIN = 2048000 - 102400
    N_VAL = 102400 

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 crop_size=None,
                 *args,
                 **kwargs):
        """
        A specific reader for Threedworld generated dataset stored as a HDF5 file

        Args:
            - data_path
                path to raw hdf5 data
        Kwargs:
            - group (str, default: 'train')
                Which subset of the dataset you want: train, val.
                The latter contains 50k images from the train set,
                so that you can directly compare performance on the validation set
                to the performance on the train set to track overfitting.
            - batch_size (int, default: 1)
                Number of images to return when `next` is called. By default set
                to 1 since it is expected to be used with queues where reading one
                image at a time is ok.
            - crop_size (int or None, default: None)
                For center crop (crop_size x crop_size). If None, no cropping will occur.
            - *args, **kwargs
                Extra arguments for HDF5DataProvider
        """
        self.group = group
        self.images = 'images'
        self.labels = 'normals'
        if self.group=='train':
            subslice = range(self.N_TRAIN)
        else:
            subslice = range(self.N_TRAIN, self.N_TRAIN + self.N_VAL)
        super(Threedworld, self).__init__(
            data_path,
            [self.images, self.labels],
            batch_size=batch_size,
            postprocess={self.images: self.postproc, self.labels: self.postproc},
            pad=True,
            subslice=subslice,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

        self.off        = None
        self.now_num    = 0

    def postproc(self, ims, f):
        norm = ims.astype(np.float32) / 255
        if self.group=='train':
            #print('In train')
            if self.now_num==0:
                off = np.random.randint(0, 256 - self.crop_size, size=2)
                self.off = off
            else:
                off = self.off
        else:
            off = int((256 - self.crop_size)/2)
            off = [off, off]
        images_batch = norm[:,
                            off[0]: off[0] + self.crop_size,
                            off[1]: off[1] + self.crop_size]
        if self.now_num==0:
            self.now_num = 1
        else:
            self.now_num = 0

        return images_batch

    def next(self):
        batch = super(Threedworld, self).next()
        feed_dict = {'images': np.squeeze(batch[self.images]),
                     'labels': np.squeeze(batch[self.labels])}
        return feed_dict

#BATCH_SIZE = 256
BATCH_SIZE = 128
NUM_BATCHES_PER_EPOCH = Threedworld.N_TRAIN // BATCH_SIZE
IMAGE_SIZE_CROP = 224
NUM_CHANNELS = 3
NORM_NUM = (IMAGE_SIZE_CROP**2) * NUM_CHANNELS * BATCH_SIZE

def loss_ave_l2(output, labels):
    loss = tf.nn.l2_loss(output - labels) / NORM_NUM
    return loss

def rep_loss(inputs, outputs, target):
    loss = tf.nn.l2_loss(outputs - inputs[target]) / NORM_NUM
    return {'loss': loss}

def postprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in ['encode', 'decode', 'hidden']:
        if k in cfg:
            ks = cfg[k].keys()
            for _k in ks:
                cfg[k][int(_k)] = cfg[k].pop(_k)
    return cfg


def preprocess_config(cfg):
    cfg = copy.deepcopy(cfg)
    for k in ['encode', 'decode', 'hidden']:
        if k in cfg:
            ks = cfg[k].keys()
            for _k in ks:
                assert isinstance(_k, int), _k
                cfg[k][str(_k)] = cfg[k].pop(_k)
    return cfg

def main(cfgfile):
    #cfg_initial = postprocess_config(json.load(open(cfgfile)))
    cfg_initial = preprocess_config(json.load(open(cfgfile)))
    params = {
        'save_params': {
            'host': 'localhost',
            #'port': 31001,
            'port': 22334,
            'dbname': 'normalnet-test',
            'collname': 'normalnet',
            'exp_id': 'trainval0',

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 5,  # keeps loss from every SAVE_LOSS_FREQ steps.
            'save_valid_freq': 3000,
            'save_filters_freq': 30000,
            'cache_filters_freq': 3000,
            # 'cache_dir': None,  # defaults to '~/.tfutils'
        },

        'load_params': {
            # 'host': 'localhost',
            # 'port': 31001,
            # 'dbname': 'alexnet-test',
            # 'collname': 'alexnet',
            # 'exp_id': 'trainval0',
            'do_restore': False,
            'load_query': None
        },

        'model_params': {
            'func': normal_encoder_asymmetric_with_bypass.normalnet_tfutils,
            'seed': 0,
            'cfg_initial': cfg_initial
        },

        'train_params': {
            'data_params': {
                'func': Threedworld,
                'data_path': DATA_PATH,
                'group': 'train',
                'crop_size': IMAGE_SIZE_CROP,
                'batch_size': 1
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE,
                'n_threads': 4,
                'seed': 0,
            },
            'thres_loss': 1000,
            'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        },

        'loss_params': {
            'targets': 'labels',
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_ave_l2
        },

        'learning_rate_params': {
            'func': tf.train.exponential_decay,
            'learning_rate': .01,
            'decay_rate': .95,
            'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
            'staircase': True
        },

        'optimizer_params': {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'clip': True,
            'momentum': .9
        },

        'validation_params': {
            'topn': {
                'data_params': {
                    'func': Threedworld,
                    'data_path': DATA_PATH,  # path to image database
                    'group': 'val',
                    'crop_size': IMAGE_SIZE_CROP,  # size after cropping an image
                },
                'targets': {
                    'func': rep_loss,
                    'target': 'labels',
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': BATCH_SIZE,
                    'n_threads': 4,
                    'seed': 0,
                },
                'num_steps': Threedworld.N_VAL // BATCH_SIZE + 1,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            },
        },

        'log_device_placement': False,  # if variable placement has to be logged
    }
    base.get_params()
    base.train_from_params(**params)


if __name__ == '__main__':
    #base.get_params()
    #base.train_from_params(**params)
    main('normals_config_winner0.cfg')
