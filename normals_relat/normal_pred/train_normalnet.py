"""
Script used for training the normalnet using tfutils
"""

from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

import normal_encoder_asymmetric_with_bypass
from sklearn.preprocessing import normalize

from tfutils import base, data, model, optimizer

import json
import copy
import argparse
#import train_normalnet_hdf5

host = os.uname()[1]

DATA_PATH = {}
if host == 'freud':  # freud
    #DATA_PATH['train'] = '/media/data/one_world_dataset/randomperm.hdf5'
    #DATA_PATH['val'] = '/media/data/one_world_dataset/randomperm_test1.hdf5'
    DATA_PATH['train/images'] = '/media/data2/one_world_dataset/tfdata/images/'
    DATA_PATH['train/normals'] = '/media/data2/one_world_dataset/tfdata/normals/'
    DATA_PATH['val/images'] = '/media/data2/one_world_dataset/tfvaldata/images/'
    DATA_PATH['val/normals'] = '/media/data2/one_world_dataset/tfvaldata/normals/'

elif host.startswith('node') and 'neuroaicluster' in host:  # OpenMind
    #DATA_PATH['train'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm.hdf5'
    #DATA_PATH['val'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm_test1.hdf5'
    #DATA_PATH['train'] = '/om/user/chengxuz/Data/one_world_dataset/dataset.tfrecords'
    #DATA_PATH['val'] = '/om/user/chengxuz/Data/one_world_dataset/dataset8.tfrecords'
    #DATA_PATH['train/images'] = '/om/user/chengxuz/Data/one_world_dataset/tfdata/images/'
    #DATA_PATH['train/normals'] = '/om/user/chengxuz/Data/one_world_dataset/tfdata/normals/'
    #DATA_PATH['val/images'] = '/om/user/chengxuz/Data/one_world_dataset/tfvaldata/images/'
    #DATA_PATH['val/normals'] = '/om/user/chengxuz/Data/one_world_dataset/tfvaldata/normals/'

    DATA_PATH['train/images'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/images/'
    DATA_PATH['train/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/normals/'
    DATA_PATH['val/images'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/images/'
    DATA_PATH['val/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/normals/'
else:
    print("Not supported yet!")
    exit()

DATA_PATH_hdf5 = {}
if host == 'freud':  # freud
    DATA_PATH_hdf5['train'] = '/media/data/one_world_dataset/randomperm.hdf5'
    DATA_PATH_hdf5['val'] = '/media/data/one_world_dataset/randomperm_test1.hdf5'
    #print("No file now!")
    #exit()

elif host.startswith('node') or host == 'openmind7':  # OpenMind
    DATA_PATH_hdf5['train'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm.hdf5'
    DATA_PATH_hdf5['val'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm_test1.hdf5'
else:
    print("Not supported yet!")
    exit()

DATA_PATH_SCENE = {}
DATA_PATH_SCENE['train/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
DATA_PATH_SCENE['train/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'
#DATA_PATH_SCENE['val/images'] = '/mnt/fs1/Dataset/scenenet_combine_val/photo'
#DATA_PATH_SCENE['val/normals'] = '/mnt/fs1/Dataset/scenenet_combine_val/normal'
DATA_PATH_SCENE['val/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
DATA_PATH_SCENE['val/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k:[] for k in res}
    for k,v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

class Threedworld_hdf5(data.ParallelBySliceProvider):

    N_TRAIN = 2048000
    N_VAL = 128000

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_size=None,
                 #center_im = False,
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
        #self.center_im = center_im

        super(Threedworld_hdf5, self).__init__(
            basefunc = data.HDF5DataReader,
            kwargs = {'hdf5source': data_path[group], 'sourcelist': [self.images, self.labels], 
                      #'postprocess':{self.images: self.postproc, self.labels: self.postproc}},
                      'postprocess':{self.images: self.postproc_image, self.labels: self.postproc_normal}},
            batch_size=batch_size,
            n_threads=n_threads,
            *args, **kwargs)

        if crop_size is None:
            self.crop_size = 224
        else:
            self.crop_size = crop_size

        self.off        = None
        self.now_num    = 0

    def get_off(self):
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

        return off

    def update_nownum(self):
        if self.now_num==0:
            self.now_num = 1
        else:
            self.now_num = 0


    def postproc_image(self, ims, f):
        norm = ims.astype(np.float32) / 255

        #if self.center_im:
        #    norm = norm - 0.5

        off = self.get_off()
        images_batch = norm[:,
                            off[0]: off[0] + self.crop_size,
                            off[1]: off[1] + self.crop_size]
        self.update_nownum()
        return images_batch

    def postproc_normal(self, ims, f):
        norm = ims.astype(np.float32) / 255

        off = self.get_off()
        images_batch = norm[:,
                            off[0]: off[0] + self.crop_size,
                            off[1]: off[1] + self.crop_size]
        self.update_nownum()
        return images_batch

class SceneNet(data.TFRecordsParallelByFileProvider):

    N_TRAIN = 5100000 
    #N_VAL = 300000
    N_VAL = 6400

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

        super(SceneNet, self).__init__(
            source_dirs = [data_path["%s/%s" % (group, self.images)] , data_path["%s/%s" % (group, self.labels)]],
            batch_size=batch_size,
            postprocess={self.images: [(self.postproc_s, (), {})], self.labels: [(self.postproc_s, (), {})]},
            trans_dicts = [{'image_raw': self.images}, {'image_raw': self.labels}], 
            n_threads=n_threads,
            shuffle = True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 224
        else:
            self.crop_size = crop_size

    def postproc_s(self, images):

        if self.group=='train':

            shape_tensor = images.get_shape().as_list()
            crop_images = tf.random_crop(images, [self.batch_size, self.crop_size, self.crop_size, shape_tensor[3]], seed=0)
            norm = tf.cast(crop_images, tf.float32)
            norm = tf.div(norm, tf.constant(255, dtype=tf.float32))

            return norm

        else:
            norm = tf.cast(images, tf.float32)
            norm = tf.div(norm, tf.constant(255, dtype=tf.float32))

            NOW_SIZE1 = 240
            NOW_SIZE2 = 320

            off = np.zeros(shape = [self.batch_size, 4])
            off[:, 0] = int((NOW_SIZE1 - self.crop_size)/2)
            off[:, 1] = int((NOW_SIZE2 - self.crop_size)/2)
            off[:, 2:4] = off[:, :2] + self.crop_size
            off[:, 0] = off[:, 0]*1.0/(NOW_SIZE1 - 1)
            off[:, 2] = off[:, 2]*1.0/(NOW_SIZE1 - 1)

            off[:, 1] = off[:, 1]*1.0/(NOW_SIZE2 - 1)
            off[:, 3] = off[:, 3]*1.0/(NOW_SIZE2 - 1)

            box_ind    = tf.constant(range(self.batch_size))

            images_batch = tf.image.crop_and_resize(norm, off, box_ind, tf.constant([self.crop_size, self.crop_size]))

            return images_batch

#class Threedworld(data.TFRecordsDataProvider):
class Threedworld(data.TFRecordsParallelByFileProvider):

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
            source_dirs = [data_path["%s/%s" % (group, self.images)] , data_path["%s/%s" % (group, self.labels)]],
            batch_size=batch_size,
            postprocess={self.images: [(self.postproc_img, (), {})], self.labels: [(self.postproc_img, (), {})]},
            #postprocess={self.images: self.postproc_img, self.labels: self.postproc_lab},
            #postprocess={self.images: self.postproc_resize, self.labels: self.postproc_resize},
            #imagelist=[self.images, self.labels],
            n_threads=n_threads,
            shuffle = True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 224
        else:
            self.crop_size = crop_size

        self.off        = None
        self.now_num    = 0

        self.box_ind    = tf.constant(range(self.batch_size))

    def postproc_resize(self, images, dtype, shape):
        norm = tf.cast(images, tf.float32)
        norm = tf.div(norm, tf.constant(255, dtype=tf.float32))
        norm = tf.cast(norm, tf.float32)

        images_batch = tf.image.resize_images(norm, tf.constant([self.crop_size, self.crop_size]))

        return [images_batch, images_batch.dtype, images_batch[0].get_shape()]

    def postproc_lab(self, images, dtype, shape):

        norm = tf.cast(images, tf.float32)
        #norm = tf.div(norm, tf.constant(255, dtype=tf.float32))
        norm = tf.nn.l2_normalize(norm, 3)
        #norm = tf.cast(norm, tf.float32)

        if self.group=='train':

            if self.now_num==0:
                off = np.zeros(shape = [self.batch_size, 4])
                off[:, :2] = np.random.randint(0, IMAGE_SIZE - self.crop_size, size=[self.batch_size, 2])
                off[:, 2:4] = off[:, :2] + self.crop_size
                off = off*1.0/(IMAGE_SIZE - 1)
                self.off = off
            else:
                off = self.off


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

    #def postproc_img(self, images, dtype, shape):
    def postproc_img(self, images):

        if self.group=='train':

            shape_tensor = images.get_shape().as_list()
            crop_images = tf.random_crop(value = images, size = [self.batch_size, self.crop_size, self.crop_size, shape_tensor[3]], seed=0)
            norm = tf.cast(crop_images, tf.float32)
            norm = tf.div(norm, tf.constant(255, dtype=tf.float32))

            return norm

        else:

            norm = tf.cast(images, tf.float32)
            norm = tf.div(norm, tf.constant(255, dtype=tf.float32))
            #norm = tf.cast(norm, tf.float32)

            off = np.zeros(shape = [self.batch_size, 4])
            off[:, :2] = int((IMAGE_SIZE - self.crop_size)/2)
            off[:, 2:4] = off[:, :2] + self.crop_size
            off = off*1.0/(IMAGE_SIZE - 1)

            box_ind    = tf.constant(range(self.batch_size))

            images_batch = tf.image.crop_and_resize(norm, off, self.box_ind, tf.constant([self.crop_size, self.crop_size]))
            return images_batch


#BATCH_SIZE = 256
#BATCH_SIZE = 192
#BATCH_SIZE = 128
#BATCH_SIZE = 64
BATCH_SIZE = 32
IMAGE_SIZE_CROP = 224
IMAGE_SIZE = 256
NUM_CHANNELS = 3
#NORM_NUM = (IMAGE_SIZE_CROP**2) * NUM_CHANNELS * BATCH_SIZE

def loss_ave_l2(output, labels):
    loss = tf.nn.l2_loss(output - labels) / np.prod(labels.get_shape().as_list())
    return loss

def loss_ave_invdot(output, labels):
    output = tf.nn.l2_normalize(output, 3)
    labels = tf.nn.l2_normalize(labels, 3)
    loss = -tf.reduce_sum(tf.multiply(output, labels)) / np.prod(labels.get_shape().as_list()) * 3
    return loss

def rep_loss(inputs, outputs, target):
    loss    = loss_ave_l2(outputs, inputs[target])
    loss_2  = loss_ave_invdot(outputs, inputs[target])
    return {'loss': loss, 'loss_2': loss_2}

def save_features(inputs, outputs, num_to_save, **loss_params):
    curr_input      = inputs['images'][:num_to_save]
    curr_input      = tf.multiply(curr_input, tf.constant(255, dtype=tf.float32))
    curr_input      = tf.cast(curr_input, tf.uint8)

    curr_output     = outputs[:num_to_save]
    curr_output     = tf.multiply(curr_output, tf.constant(255, dtype=tf.float32))
    curr_output     = tf.cast(curr_output, tf.uint8)

    curr_label      = inputs['normals'][:num_to_save]
    curr_label      = tf.multiply(curr_label, tf.constant(255, dtype=tf.float32))
    curr_label      = tf.cast(curr_label, tf.uint8)
    #curr_label      = tf.cast(curr_output, tf.uint8)

    #loss    = tf.nn.l2_loss(outputs - inputs['normals']) / NORM_NUM

    #return {'loss': loss, 'images': curr_input, 'normals': curr_label, 'outputs': curr_output}
    return {'images_fea': curr_input, 'normals_fea': curr_label, 'outputs_fea': curr_output}

def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    for k in keys:
        plucked = [d[k] for d in step_results]
        if 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval

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
    #cfg_initial = preprocess_config(json.load(open(args.pathconfig)))
    cfg_initial = postprocess_config(json.load(open(args.pathconfig)))
    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)

    #queue_capa = BATCH_SIZE*120
    #queue_capa = BATCH_SIZE*500
    BATCH_SIZE  = normal_encoder_asymmetric_with_bypass.getBatchSize(cfg_initial)
    if args.batchsize:
        BATCH_SIZE = args.batchsize
    queue_capa  = normal_encoder_asymmetric_with_bypass.getQueueCap(cfg_initial)
    n_threads   = 4

    func_net = getattr(normal_encoder_asymmetric_with_bypass, args.namefunc)

    train_data_param = {
                'func': Threedworld_hdf5,
                #'func': train_normalnet_hdf5.Threedworld,
                'data_path': DATA_PATH_hdf5,
                'group': 'train',
                'crop_size': IMAGE_SIZE_CROP,
                'n_threads': n_threads,
                'batch_size': 2,
            }
    val_data_param = {
                    'func': Threedworld_hdf5,
                    #'func': train_normalnet_hdf5.Threedworld,
                    'data_path': DATA_PATH_hdf5,
                    'group': 'val',
                    'crop_size': IMAGE_SIZE_CROP,
                    'n_threads': n_threads,
                    'batch_size': 2,
                }
    train_queue_params = {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE,
                'seed': 0,
                'capacity': BATCH_SIZE*10,
            }
    val_queue_params    = train_queue_params
    val_target          = 'normals'

    if args.usehdf5==0:
        train_data_param['func']   = Threedworld
        val_data_param['func']     = Threedworld
        train_data_param['data_path']   = DATA_PATH
        val_data_param['data_path']   = DATA_PATH
        #train_data_param['n_threads'] = n_threads
        #val_data_param['n_threads'] = n_threads

        train_queue_params = {
                'queue_type': 'random',
                'batch_size': BATCH_SIZE,
                'seed': 0,
                'capacity': queue_capa,
                # 'n_threads' : 4
            }
        val_queue_params = {
                    'queue_type': 'fifo',
                    'batch_size': BATCH_SIZE,
                    'seed': 0,
                    'capacity': BATCH_SIZE*10,
                }
        val_target          = 'normals'

    if args.whichdataset==1:
        train_data_param['func']   = SceneNet
        val_data_param['func']     = SceneNet
        train_data_param['data_path']   = DATA_PATH_SCENE
        val_data_param['data_path']   = DATA_PATH_SCENE


    val_step_num = val_data_param['func'].N_VAL // BATCH_SIZE + 1
    NUM_BATCHES_PER_EPOCH = train_data_param['func'].N_TRAIN // BATCH_SIZE

    if args.valinum>-1:
        val_step_num = args.valinum

    loss_func = loss_ave_l2
    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': .01,
            'decay_rate': .95,
            'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
            'staircase': True
        }

    model_params = {
            'func': func_net,
            'seed': args.seed,
            'cfg_initial': cfg_initial
        }

    optim_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'clip': True,
            'momentum': .9
        }

    if args.whichloss==1:
        loss_func = loss_ave_invdot
        learning_rate_params = {
                'func': tf.train.exponential_decay,
                'learning_rate': .001,
                'decay_rate': .5,
                'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                'staircase': True
            }
        #optimizer_class     = tf.train.RMSPropOptimizer
        #train_data_param['center_im'] = True
        #val_data_param['center_im'] = True
        model_params['center_im']   = True
        optim_params = {
                'func': optimizer.ClipOptimizer,
                'optimizer_class': tf.train.RMSPropOptimizer,
                'clip': True,
            }

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
            'save_valid_freq': 5000,
            #'save_metrics_freq': 100,  # keeps loss from every SAVE_LOSS_FREQ steps.
            #'save_valid_freq': 100,
            'save_filters_freq': 5000,
            'cache_filters_freq': 5000,
            'cache_dir': cache_dir,  # defaults to '~/.tfutils'
            'save_to_gfs': ['images_fea', 'normals_fea', 'outputs_fea'], 
            #'save_intermediate_freq': 1,
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

        'model_params': model_params,

        'train_params': {
            #'validate_first': False,
            'validate_first': True,
            'data_params': train_data_param,
            'queue_params': train_queue_params,
            'thres_loss': 1000,
            'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        },

        'loss_params': {
            'targets': val_target,
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_func,
            'loss_per_case_func_params': {}
        },

        'learning_rate_params': learning_rate_params,

        'optimizer_params': optim_params,
        'log_device_placement': False,  # if variable placement has to be logged
        'validation_params': {
            'topn': {
                'data_params': val_data_param,
                'queue_params': val_queue_params,
                'targets': {
                    'func': rep_loss,
                    'target': val_target,
                },
                #'num_steps': Threedworld.N_VAL // BATCH_SIZE + 1,
                'num_steps': val_step_num,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            },
            'feats':{
                'data_params': val_data_param,
                'queue_params': val_queue_params,
                'targets': {
                    'func': save_features,
                    'num_to_save': 5,
                    'targets' : [],
                },
                #'num_steps': Threedworld.N_VAL // BATCH_SIZE + 1,
                'num_steps': 10,
                'agg_func': mean_losses_keep_rest,
                #'online_agg_func': online_agg
            },
        },
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
    parser.add_argument('--namefunc', default = "normalnet_tfutils", type = str, action = 'store', help = 'Name of function to build the network')
    parser.add_argument('--usehdf5', default = 0, type = int, action = 'store', help = 'Whether use hdf5 data reader')
    parser.add_argument('--valinum', default = -1, type = int, action = 'store', help = 'Number of validation steps, default is -1, which means all the validation')
    parser.add_argument('--whichloss', default = 0, type = int, action = 'store', help = 'Whether to use new loss')
    parser.add_argument('--whichdataset', default = 0, type = int, action = 'store', help = '0 for threedworld, 1 for scenenet')
    parser.add_argument('--batchsize', default = None, type = int, action = 'store', help = 'None for default')

    args    = parser.parse_args()

    main(args)
