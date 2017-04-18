from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse

sys.path.append('../normal_pred/')
import normal_encoder_asymmetric_with_bypass
import combinet_builder

host = os.uname()[1]

DATA_PATH = {}

if 'neuroaicluster' in host:
    DATA_PATH['threed/train/images'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/images'
    DATA_PATH['threed/train/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfdata/normals'
    
    DATA_PATH['threed/val/images'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/images'
    DATA_PATH['threed/val/normals'] = '/mnt/fs0/datasets/one_world_dataset/tfvaldata/normals'

    DATA_PATH['scenenet/train/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
    DATA_PATH['scenenet/train/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'

    #DATA_PATH['scenenet/val/images'] = '/mnt/fs1/Dataset/scenenet_combine_val/photo'
    #DATA_PATH['scenenet/val/normals'] = '/mnt/fs1/Dataset/scenenet_combine_val/normal'
    DATA_PATH['scenenet/val/images'] = '/mnt/fs1/Dataset/scenenet_combine/photo'
    DATA_PATH['scenenet/val/normals'] = '/mnt/fs1/Dataset/scenenet_combine/normal_new'

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

class Combine_world(data.TFRecordsParallelByFileProvider):

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_size=None,
                 only_one=None,
                 *args, **kwargs
                 ):
        self.group = group
        self.batch_size = batch_size

        # Keys for threedworld
        self.image_t = 'image_t'
        self.normal_t = 'normal_t'

        # Keys for scenenet
        self.image_s = 'image_s'
        self.normal_s = 'normal_s'

        self.crop_size = 224
        if not crop_size==None:
            self.crop_size = crop_size

        if only_one==None:
            postprocess = {self.image_t: [(self.postproc_t, (), {})], self.normal_t: [(self.postproc_t, (), {})], 
                     self.image_s: [(self.postproc_s, (), {})], self.normal_s: [(self.postproc_s, (), {})]}

            super(Combine_world, self).__init__(
                source_dirs = [data_path["threed/%s/images" % group] , data_path["threed/%s/normals" % group] , data_path["scenenet/%s/images" % group], data_path["scenenet/%s/normals" % group]],
                trans_dicts = [{'images': self.image_t}, {'normals': self.normal_t}, {'image_raw': self.image_s}, {'image_raw': self.normal_s}], 
                postprocess = postprocess,
                batch_size=batch_size,
                n_threads=n_threads,
                shuffle = True,
                *args, **kwargs)
        elif only_one==0: # initialize for threedworld
            postprocess = {self.image_t: [(self.postproc_t, (), {})], self.normal_t: [(self.postproc_t, (), {})]}

            super(Combine_world, self).__init__(
                source_dirs = [data_path["threed/%s/images" % group] , data_path["threed/%s/normals" % group]],
                trans_dicts = [{'images': self.image_t}, {'normals': self.normal_t}], 
                postprocess = postprocess,
                batch_size=batch_size,
                n_threads=n_threads,
                shuffle = True,
                *args, **kwargs)
        elif only_one==1: # initialize for scenenet
            postprocess = {self.image_s: [(self.postproc_s, (), {})], self.normal_s: [(self.postproc_s, (), {})]}

            super(Combine_world, self).__init__(
                source_dirs = [data_path["scenenet/%s/images" % group], data_path["scenenet/%s/normals" % group]],
                trans_dicts = [{'image_raw': self.image_s}, {'image_raw': self.normal_s}], 
                postprocess = postprocess,
                batch_size=batch_size,
                n_threads=n_threads,
                shuffle = True,
                *args, **kwargs)

    def postproc_s(self, images):

        norm = tf.cast(images, tf.float32)
        norm = tf.div(norm, tf.constant(255, dtype=tf.float32))

        return self.postproc_flag(norm, 1)

    def postproc_t(self, images):

        norm = tf.cast(images, tf.float32)
        norm = tf.div(norm, tf.constant(255, dtype=tf.float32))

        return self.postproc_flag(norm, 0)

    def postproc_flag(self, norm, flag):

        if flag==0:
            NOW_SIZE1 = 256
            NOW_SIZE2 = 256
            seed_random = 0

        if flag==1:
            NOW_SIZE1 = 240
            NOW_SIZE2 = 320
            seed_random = 1

        if self.group=='train':

            shape_tensor = norm.get_shape().as_list()
            crop_images = tf.random_crop(norm, [self.batch_size, self.crop_size, self.crop_size, shape_tensor[3]], seed=seed_random)

            return crop_images

        else:

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
        
class Combine_world_sep:

    def __init__(self, *args, **kwargs):
        self.data_t = Combine_world(only_one = 0, *args, **kwargs)
        self.data_s = Combine_world(only_one = 1, *args, **kwargs)

    def init_ops(self):
        self.init_ops_t = self.data_t.init_ops()
        self.init_ops_s = self.data_s.init_ops()
        num_threads = len(self.init_ops_t)

        self.init_ops = []

        for indx_t in xrange(num_threads):
            tmp_op = self.init_ops_t[indx_t]
            tmp_op.update(self.init_ops_s[indx_t])
            self.init_ops.append(tmp_op)

        return self.init_ops


BATCH_SIZE = 32
IMAGE_SIZE_CROP = 224
NUM_CHANNELS = 3

def loss_ave_l2(output, label_0, label_1):
    loss_0 = tf.nn.l2_loss(output[0] - label_0) / np.prod(label_0.get_shape().as_list())
    loss_1 = tf.nn.l2_loss(output[1] - label_1) / np.prod(label_1.get_shape().as_list())
    loss = tf.add(loss_0, loss_1)
    return loss

def loss_ave_invdot(output, label_0, label_1):
    output_0 = tf.nn.l2_normalize(output[0], 3)
    labels_0 = tf.nn.l2_normalize(label_0, 3)
    loss_0 = -tf.reduce_sum(tf.multiply(output_0, labels_0)) / np.prod(label_0.get_shape().as_list()) * 3

    output_1 = tf.nn.l2_normalize(output[1], 3)
    labels_1 = tf.nn.l2_normalize(label_1, 3)
    loss_1 = -tf.reduce_sum(tf.multiply(output_1, labels_1)) / np.prod(label_1.get_shape().as_list()) * 3

    loss = tf.add(loss_0, loss_1)
    return loss

def rep_loss(inputs, outputs, target):
    loss    = loss_ave_l2(outputs, inputs[target[0]], inputs[target[1]])
    loss_2  = loss_ave_invdot(outputs, inputs[target[0]], inputs[target[1]])
    return {'loss': loss, 'loss_2': loss_2}

def save_features(inputs, outputs, num_to_save, **loss_params):
    curr_input_t      = inputs['image_t'][:num_to_save]
    curr_input_t      = tf.multiply(curr_input_t, tf.constant(255, dtype=tf.float32))
    curr_input_t      = tf.cast(curr_input_t, tf.uint8)

    curr_input_s      = inputs['image_s'][:num_to_save]
    curr_input_s      = tf.multiply(curr_input_s, tf.constant(255, dtype=tf.float32))
    curr_input_s      = tf.cast(curr_input_s, tf.uint8)

    curr_output_0     = outputs[0][:num_to_save]
    curr_output_0     = tf.multiply(curr_output_0, tf.constant(255, dtype=tf.float32))
    curr_output_0     = tf.cast(curr_output_0, tf.uint8)

    curr_output_1     = outputs[1][:num_to_save]
    curr_output_1     = tf.multiply(curr_output_1, tf.constant(255, dtype=tf.float32))
    curr_output_1     = tf.cast(curr_output_1, tf.uint8)

    curr_label_t      = inputs['normal_t'][:num_to_save]
    curr_label_t      = tf.multiply(curr_label_t, tf.constant(255, dtype=tf.float32))
    curr_label_t      = tf.cast(curr_label_t, tf.uint8)

    curr_label_s      = inputs['normal_s'][:num_to_save]
    curr_label_s      = tf.multiply(curr_label_s, tf.constant(255, dtype=tf.float32))
    curr_label_s      = tf.cast(curr_label_s, tf.uint8)

    return {'images_fea_t': curr_input_t, 'normals_fea_t': curr_label_t, 'outputs_fea_t': curr_output_0, 
            'images_fea_s': curr_input_s, 'normals_fea_s': curr_label_s, 'outputs_fea_s': curr_output_1}

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

def main():
    parser = argparse.ArgumentParser(description='The script to train the combine net')
    parser.add_argument('--nport', default = 29101, type = int, action = 'store', help = 'Port number of mongodb')
    parser.add_argument('--pathconfig', default = "normals_config_fcnvgg16.cfg", type = str, action = 'store', help = 'Path to config file')
    parser.add_argument('--expId', default = "combinet_test", type = str, action = 'store', help = 'Name of experiment id')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed for model')
    parser.add_argument('--gpu', default = -1, type = int, action = 'store', help = 'Index of gpu, currently only one gpu is allowed')
    parser.add_argument('--cacheDirPrefix', default = "/mnt/fs1/chengxuz", type = str, action = 'store', help = 'Prefix of cache directory')
    parser.add_argument('--namefunc', default = "combine_normal_tfutils", type = str, action = 'store', help = 'Name of function to build the network')
    parser.add_argument('--valinum', default = -1, type = int, action = 'store', help = 'Number of validation steps, default is -1, which means all the validation')
    parser.add_argument('--whichloss', default = 0, type = int, action = 'store', help = 'Whether to use new loss')

    args    = parser.parse_args()

    if args.gpu>-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg_initial = postprocess_config(json.load(open(args.pathconfig)))
    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)

    BATCH_SIZE  = normal_encoder_asymmetric_with_bypass.getBatchSize(cfg_initial)
    queue_capa  = normal_encoder_asymmetric_with_bypass.getQueueCap(cfg_initial)
    n_threads   = 4

    func_net = getattr(combinet_builder, args.namefunc)

    train_data_param = {
                #'func': Combine_world,
                'func': Combine_world_sep,
                'data_path': DATA_PATH,
                'group': 'train',
                'n_threads': n_threads,
                'batch_size': 2,
            }
    val_data_param = {
                #'func': Combine_world,
                'func': Combine_world_sep,
                'data_path': DATA_PATH,
                'group': 'val',
                'n_threads': 1,
                'batch_size': 2,
            }
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
    val_target          = ['normal_t', 'normal_s']

    val_step_num = 500
    NUM_BATCHES_PER_EPOCH = 5000

    if args.valinum>-1:
        val_step_num = args.valinum

    loss_func = loss_ave_l2
    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': .01,
            'decay_rate': .95,
            'decay_steps': 2*NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
            'staircase': True
        }

    optimizer_class = tf.train.MomentumOptimizer

    model_params = {
            'func': func_net,
            'seed': args.seed,
            'cfg_initial': cfg_initial
        }

    if args.whichloss==1:
        loss_func = loss_ave_invdot
        learning_rate_params = {
                'func': tf.train.exponential_decay,
                'learning_rate': .001,
                'decay_rate': .5,
                'decay_steps': 5*NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
                'staircase': True
            }

        model_params['center_im']   = True

    dbname = 'combinet-test'
    collname = 'combinet'

    params = {
        'save_params': {
            'host': 'localhost',
            'port': args.nport,
            'dbname': dbname,
            'collname': collname,
            'exp_id': exp_id,

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 2000,  # keeps loss from every SAVE_LOSS_FREQ steps.
            'save_valid_freq': 5000,
            'save_filters_freq': 5000,
            'cache_filters_freq': 5000,
            'cache_dir': cache_dir,  # defaults to '~/.tfutils'
            'save_to_gfs': ['images_fea_t', 'normals_fea_t', 'outputs_fea_t', 
                'images_fea_s', 'normals_fea_s', 'outputs_fea_s'],
        },

        'load_params': {
            'host': 'localhost',
            'port': args.nport,
            'dbname': dbname,
            'collname': collname,
            'exp_id': exp_id,
            'do_restore': True,
            'load_query': None
        },

        'model_params': model_params,

        'train_params': {
            'validate_first': False,
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

        'optimizer_params': {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': optimizer_class,
            'clip': True,
            'momentum': .9
        },
        'log_device_placement': False,  # if variable placement has to be logged
        'validation_params': {
            'topn': {
                'data_params': val_data_param,
                'queue_params': val_queue_params,
                'targets': {
                    'func': rep_loss,
                    'target': val_target,
                },
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
                'num_steps': 10,
                'agg_func': mean_losses_keep_rest,
            },
        },
    }
    base.train_from_params(**params)

if __name__ == '__main__':
    main()

'''
'''
