from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

from tfutils import base, data, optimizer

import json
import copy
import argparse

import cate_network_builder

host = os.uname()[1]

DATA_PATH = {}
DATA_PATH['train/Data_force'] = '/media/data3/chengxuz/whisker/tfrecords/Data_force/'
DATA_PATH['train/Data_torque'] = '/media/data3/chengxuz/whisker/tfrecords/Data_torque/'
DATA_PATH['train/category'] = '/media/data3/chengxuz/whisker/tfrecords/category/'
#DATA_PATH['val/images'] = '/media/data2/one_world_dataset/tfvaldata/images/'
#DATA_PATH['val/normals'] = '/media/data2/one_world_dataset/tfvaldata/normals/'

if 'neuroaicluster' in host:
    DATA_PATH['train/Data_force'] = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords/Data_force/'
    DATA_PATH['train/Data_torque'] = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords/Data_torque/'
    DATA_PATH['train/category'] = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords/category/'
    DATA_PATH['train/trainflag'] = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords/trainflag/'
    DATA_PATH['val/Data_force'] = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val/Data_force/'
    DATA_PATH['val/Data_torque'] = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val/Data_torque/'
    DATA_PATH['val/category'] = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val/category/'
    DATA_PATH['val/trainflag'] = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val/trainflag/'

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def in_top_k(inputs, outputs, target):
    return {'top1': tf.nn.in_top_k(outputs, inputs[target], 1),
            'top5': tf.nn.in_top_k(outputs, inputs[target], 5)}

class WhiskerWorld(data.TFRecordsParallelByFileProvider):

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 n_threads=4,
                 crop_size=None,
                 *args,
                 **kwargs):

        self.group = group
        self.force = 'Data_force'
        self.torque = 'Data_torque'
        self.label = 'category'
        self.trainflag = 'trainflag'
        self.batch_size = batch_size
        postprocess = {self.force: [(self.postprocess_images, (), {})], self.torque: [(self.postprocess_images, (), {})]}

        super(WhiskerWorld, self).__init__(
            source_dirs = [data_path["%s/%s" % (group, self.force)] , data_path["%s/%s" % (group, self.torque)] , data_path["%s/%s" % (group, self.label)], data_path["%s/%s" % (group, self.trainflag)]],
            postprocess = postprocess,
            batch_size=batch_size,
            n_threads=n_threads,
            shuffle = True,
            *args, **kwargs)

    def set_data_shapes(self, data):
        for i in range(len(data)):
            for k in data[i]:
                # set shape[0] to batch size for all entries
                shape = data[i][k].get_shape().as_list()
                shape[0] = self.batch_size
                data[i][k].set_shape(shape)
        return data

    def slice_concat(self, data, curr_key, new_key):
        #print(data[curr_key].get_shape().as_list())
        slice0 = tf.strided_slice( data[curr_key], [0,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice1 = tf.strided_slice( data[curr_key], [1,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice2 = tf.strided_slice( data[curr_key], [2,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        #slice1 = tf.strided_slice( data[curr_key], [1], [2], [3])
        #slice2 = tf.strided_slice( data[curr_key], [2], [3], [3])
        #print(slice0.get_shape().as_list())
        data[new_key] = tf.concat([slice0, slice1, slice2], 1)
        #print(data[new_key].get_shape().as_list())

        return data

    def slice_label(self, data, curr_key, new_key):
        #print(data[curr_key].get_shape().as_list())
        slice0 = tf.strided_slice( data[curr_key], [0], [0], [3], end_mask = 1)
        data[new_key] = slice0

        return data

    def init_ops(self):
        self.input_ops = super(WhiskerWorld, self).init_ops()

        # make sure batch size shapes of tensors are set
        self.input_ops = self.set_data_shapes(self.input_ops)

        for i in range(len(self.input_ops)):
            self.input_ops[i] = self.slice_concat(self.input_ops[i], 'Data_force', 'Data_force')
            self.input_ops[i] = self.slice_concat(self.input_ops[i], 'Data_torque', 'Data_torque')
            self.input_ops[i] = self.slice_label(self.input_ops[i], 'category', 'category')
            self.input_ops[i] = self.slice_label(self.input_ops[i], 'trainflag', 'trainflag')

        return self.input_ops

    def postprocess_images(self, ims):
        def _postprocess_images(im):
            im = tf.decode_raw(im, np.float32)
            im = tf.reshape(im, [110, 31, 3, 3])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

def main():
    parser = argparse.ArgumentParser(description='The script to train the catenet for barrel')
    parser.add_argument('--nport', default = 29101, type = int, action = 'store', help = 'Port number of mongodb')
    parser.add_argument('--pathconfig', default = "catenet_config.cfg", type = str, action = 'store', help = 'Path to config file')
    parser.add_argument('--expId', default = "catenet", type = str, action = 'store', help = 'Name of experiment id')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed for model')
    parser.add_argument('--gpu', default = -1, type = int, action = 'store', help = 'Index of gpu, currently only one gpu is allowed')
    parser.add_argument('--cacheDirPrefix', default = "/media/data2/chengxuz", type = str, action = 'store', help = 'Prefix of cache directory')
    parser.add_argument('--namefunc', default = "catenet_tfutils", type = str, action = 'store', help = 'Name of function to build the network')
    parser.add_argument('--valinum', default = -1, type = int, action = 'store', help = 'Number of validation steps, default is -1, which means all the validation')
    parser.add_argument('--whichopt', default = 0, type = int, action = 'store', help = 'Choice of the optimizer, 0 means momentum, 1 means Adam')
    parser.add_argument('--initlr', default = 0.0001, type = float, action = 'store', help = 'Initial learning rate')

    args    = parser.parse_args()

    if args.gpu>-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg_initial = json.load(open(args.pathconfig))
    #print(cfg_initial)

    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)

    BATCH_SIZE  = cfg_initial["BATCH_SIZE"]
    queue_capa  = cfg_initial["QUEUE_CAP"]
    n_threads   = 4

    func_net = getattr(cate_network_builder, args.namefunc)

    train_data_param = {
                'func': WhiskerWorld,
                'data_path': DATA_PATH,
                'group': 'train',
                'n_threads': n_threads,
                'batch_size': BATCH_SIZE,
            }
    val_data_param = {
                    'func': WhiskerWorld,
                    'data_path': DATA_PATH,
                    'group': 'val',
                    'n_threads': n_threads,
                    'batch_size': BATCH_SIZE,
                }

    train_queue_params = {
            'queue_type': 'random',
            'batch_size': BATCH_SIZE//3,
            'seed': 0,
            'capacity': queue_capa//3,
        }
    val_queue_params = {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE//3,
                'seed': 0,
                'capacity': BATCH_SIZE*10//3,
            }
    val_target          = 'category'

    val_step_num = 12*2*9981//BATCH_SIZE
    NUM_BATCHES_PER_EPOCH = 12*24*9981//BATCH_SIZE

    if args.valinum>-1:
        val_step_num = args.valinum

    loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
    learning_rate_params = {
            'func': tf.train.exponential_decay,
            'learning_rate': args.initlr,
            'decay_rate': .5,
            'decay_steps': NUM_BATCHES_PER_EPOCH*10,  # exponential decay each epoch
            'staircase': True
        }

    optimizer_class = tf.train.MomentumOptimizer

    model_params = {
            'func': func_net,
            'seed': args.seed,
            'cfg_initial': cfg_initial
        }

    optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': optimizer_class,
            'clip': True,
            'momentum': .9
        }

    if args.whichopt==1:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdamOptimizer,
            'clip': True,
        }

    if args.whichopt==2:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdagradOptimizer,
            'clip': True,
        }

    if args.whichopt==3:
        optimizer_params = {
                'func': optimizer.ClipOptimizer,
                'optimizer_class': optimizer_class,
                'clip': True,
                'momentum': .9,
                'use_nesterov': True
            }

    params = {
        'save_params': {
            'host': 'localhost',
            'port': args.nport,
            'dbname': 'whisker_net',
            'collname': 'catenet',
            'exp_id': exp_id,

            'do_save': True,
            'save_initial_filters': True,
            'save_metrics_freq': 2000,  # keeps loss from every SAVE_LOSS_FREQ steps.
            'save_valid_freq': 5000,
            'save_filters_freq': 5000,
            'cache_filters_freq': 5000,
            'cache_dir': cache_dir,
        },

        'load_params': {
            'host': 'localhost',
            'port': args.nport,
            'dbname': 'whisker_net',
            'collname': 'catenet',
            'exp_id': exp_id,
            'do_restore': True,
            'load_query': None
        },

        'model_params': model_params,

        'train_params': {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': train_queue_params,
            'thres_loss': 1000000,
            'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        },

        'loss_params': {
            'targets': val_target,
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_func,
        },

        'learning_rate_params': learning_rate_params,

        'optimizer_params': optimizer_params,

        'log_device_placement': False,  # if variable placement has to be logged
        'validation_params': {
            'topn': {
                'data_params': val_data_param,
                'queue_params': val_queue_params,
                'targets': {
                    'func': in_top_k,
                    'target': val_target,
                },
                'num_steps': val_step_num,
                'agg_func': lambda x: {k:np.mean(v) for k,v in x.items()},
                'online_agg_func': online_agg
            }
        },
    }
    #base.get_params()
    base.train_from_params(**params)

if __name__ == '__main__':
    main()
