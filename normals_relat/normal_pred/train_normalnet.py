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
import argparse

from tfutils.data import TFRecordsDataProvider#, LMDBDataProvider

#os.environ["CUDA_VISIBLE_DEVICES"]="2"

host = os.uname()[1]

DATA_PATH = {}
if host == 'freud':  # freud
    #DATA_PATH['train'] = '/media/data/one_world_dataset/randomperm.hdf5'
    #DATA_PATH['val'] = '/media/data/one_world_dataset/randomperm_test1.hdf5'
    DATA_PATH['train'] = '/media/data2/one_world_dataset/dataset.tfrecords'
    DATA_PATH['val'] = '/media/data2/one_world_dataset/dataset8.tfrecords'

elif host.startswith('node') or host == 'openmind7':  # OpenMind
    #DATA_PATH['train'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm.hdf5'
    #DATA_PATH['val'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm_test1.hdf5'
    DATA_PATH['train'] = '/om/user/chengxuz/Data/one_world_dataset/dataset.tfrecords'
    DATA_PATH['val'] = '/om/user/chengxuz/Data/one_world_dataset/dataset8.tfrecords'
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


class Threedworld(TFRecordsDataProvider):

    N_TRAIN = 2048000
    N_VAL = 128000

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 n_threads=4,
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

        super(Threedworld, self).__init__(
            data_path[group],
            {self.images: tf.string, self.labels: tf.string},
            batch_size=batch_size,
            #postprocess={self.images: self.postproc, self.labels: self.postproc},
            postprocess={self.images: self.postproc_resize, self.labels: self.postproc_resize},
            imagelist=[self.images, self.labels],
            n_threads=n_threads,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 224
        else:
            self.crop_size = crop_size

        self.off        = None
        self.now_num    = 0

        self.box_ind    = tf.constant(range(self.batch_size))

    def postproc_resize(self, images, dtype, shape):
        norm = tf.div(images, tf.constant(255, dtype=images.dtype))
        norm = tf.cast(norm, tf.float32)

        images_batch = tf.image.resize_images(norm, tf.constant([self.crop_size, self.crop_size]))

        return [images_batch, images_batch.dtype, images_batch[0].get_shape()]

    def postproc(self, images, dtype, shape):

        norm = tf.div(images, tf.constant(255, dtype=images.dtype))
        norm = tf.cast(norm, tf.float32)
        if self.group=='train':

            '''
            if self.now_num==0:
                off = np.zeros(shape = [self.batch_size, 4])
                off[:, :2] = np.random.randint(0, IMAGE_SIZE - self.crop_size, size=[self.batch_size, 2])
                off[:, 2:4] = off[:, :2] + self.crop_size
                off = off*1.0/(IMAGE_SIZE - 1)
                self.off = off
            else:
                off = self.off
            '''
            off = np.zeros(shape = [self.batch_size, 4])
            off[:, :2] = int((IMAGE_SIZE - self.crop_size)/2)
            off[:, 2:4] = off[:, :2] + self.crop_size
            off = off*1.0/(IMAGE_SIZE - 1)

        else:
            off = np.zeros(shape = [self.batch_size, 4])
            off[:, :2] = int((IMAGE_SIZE - self.crop_size)/2)
            off[:, 2:4] = off[:, :2] + self.crop_size
            off = off*1.0/(IMAGE_SIZE - 1)

        images_batch = tf.image.crop_and_resize(norm, off, self.box_ind, tf.constant([self.crop_size, self.crop_size]))
        if self.now_num==0:
            self.now_num = 1
        else:
            self.now_num = 0

        return [images_batch, images_batch.dtype, images_batch[0].get_shape()]

    def init_threads(self):
        self.input_ops, self.dtypes, self.shapes = \
                super(Threedworld, self).init_threads()

        return [self.input_ops, self.dtypes, self.shapes]


#BATCH_SIZE = 256
#BATCH_SIZE = 192
BATCH_SIZE = 128
NUM_BATCHES_PER_EPOCH = Threedworld.N_TRAIN // BATCH_SIZE
IMAGE_SIZE_CROP = 224
IMAGE_SIZE = 256
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
                #assert isinstance(_k, int), _k
                cfg[k][str(_k)] = cfg[k].pop(_k)
    return cfg

def main(args):
    #cfg_initial = postprocess_config(json.load(open(cfgfile)))
    if args.gpu>-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg_initial = preprocess_config(json.load(open(args.pathconfig)))
    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)

    queue_capa = BATCH_SIZE*120
    n_threads = 1

    params = {
        'save_params': {
            'host': 'localhost',
            #'port': 31001,
            'port': args.nport,
            'dbname': 'normalnet-test',
            'collname': 'normalnet',
            #'exp_id': 'trainval0',
            'exp_id': exp_id,
            #'exp_id': 'trainval2', # using screen?

            'do_save': True,
            #'do_save': False,
            'save_initial_filters': True,
            'save_metrics_freq': 2000,  # keeps loss from every SAVE_LOSS_FREQ steps.
            'save_valid_freq': 10000,
            'save_filters_freq': 30000,
            'cache_filters_freq': 10000,
            'cache_dir': cache_dir,  # defaults to '~/.tfutils'
        },

        'load_params': {
            'host': 'localhost',
            # 'port': 31001,
            # 'dbname': 'alexnet-test',
            # 'collname': 'alexnet',
            # 'exp_id': 'trainval0',
            'port': args.nport,
            'dbname': 'normalnet-test',
            'collname': 'normalnet',
            #'exp_id': 'trainval0',
            'exp_id': exp_id,
            #'exp_id': 'trainval2', # using screen?
            'do_restore': True,
            'load_query': None
        },

        'model_params': {
            'func': normal_encoder_asymmetric_with_bypass.normalnet_tfutils,
            'seed': args.seed,
            'cfg_initial': cfg_initial
        },

        'train_params': {
            'data_params': {
                'func': Threedworld,
                'data_path': DATA_PATH,
                'group': 'train',
                'crop_size': IMAGE_SIZE_CROP,
                'batch_size': BATCH_SIZE,
                'n_threads' : n_threads
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': BATCH_SIZE,
                'seed': 0,
                'capacity': queue_capa,
                # 'n_threads' : 4
            },
            'thres_loss': 1000,
            'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        },

        'loss_params': {
            'targets': 'normals',
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_ave_l2,
            'loss_per_case_func_params': {}
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
        'log_device_placement': False,  # if variable placement has to be logged
    }
    #base.get_params()
    base.train_from_params(**params)

if __name__ == '__main__':
    #base.get_params()
    #base.train_from_params(**params)
    parser = argparse.ArgumentParser(description='The script to train the normalnet')
    parser.add_argument('--nport', default = 22334, type = int, action = 'store', help = 'Port number of mongodb')
    parser.add_argument('--pathconfig', default = "normals_config_winner0.cfg", type = str, action = 'store', help = 'Path to config file')
    parser.add_argument('--expId', default = "trainval2", type = str, action = 'store', help = 'Name of experiment id')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed for model')
    parser.add_argument('--gpu', default = -1, type = int, action = 'store', help = 'Index of gpu, currently only one gpu is allowed')
    parser.add_argument('--cacheDirPrefix', default = "/home/chengxuz", type = str, action = 'store', help = 'Prefix of cache directory')

    args    = parser.parse_args()

    main(args)

''' 
        'validation_params': {
            'topn': {
                'data_params': {
                    'func': Threedworld,
                    'data_path': DATA_PATH,
                    'group': 'val',
                    'crop_size': IMAGE_SIZE_CROP,
                    'batch_size': BATCH_SIZE,
                    'n_threads' : n_threads
                },
                'queue_params': {
                    'queue_type': 'random',
                    'batch_size': BATCH_SIZE,
                    'seed': 0,
                    'capacity': queue_capa,
                },
                'targets': {
                    'func': rep_loss,
                    'target': 'normals',
                },
                'num_steps': Threedworld.N_VAL // BATCH_SIZE + 1,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            },
        },

'''
