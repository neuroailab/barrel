from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

from tfutils import base, data, optimizer

import json
import copy
import argparse

import cate_network_builder
import h5py

host = os.uname()[1]

DATA_PATH = {}
DATA_PATH['train/Data_force'] = '/media/data3/chengxuz/whisker/tfrecords/Data_force/'
DATA_PATH['train/Data_torque'] = '/media/data3/chengxuz/whisker/tfrecords/Data_torque/'
DATA_PATH['train/category'] = '/media/data3/chengxuz/whisker/tfrecords/category/'
#DATA_PATH['val/images'] = '/media/data2/one_world_dataset/tfvaldata/images/'
#DATA_PATH['val/normals'] = '/media/data2/one_world_dataset/tfvaldata/normals/'

train_data_path_prefix = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords'
val_data_path_prefix = '/mnt/fs0/chengxuz/Data/whisker/val_tfrecs/tfrecords_val'
#train_data_path_prefix = '/data/chengxuz/whisker/tfrecs_all/tfrecords'
#val_data_path_prefix = '/data/chengxuz/whisker/val_tfrecs/tfrecords_val'

save_num_now = None

if 'neuroaicluster' in host:
    DATA_PATH['train/Data_force'] = train_data_path_prefix + '/Data_force/'
    DATA_PATH['train/Data_torque'] = train_data_path_prefix + '/Data_torque/'
    DATA_PATH['train/category'] = train_data_path_prefix + '/category/'
    DATA_PATH['val/Data_force'] = val_data_path_prefix + '/Data_force/'
    DATA_PATH['val/Data_torque'] = val_data_path_prefix + '/Data_torque/'
    DATA_PATH['val/category'] = val_data_path_prefix + '/category/'
    DATA_PATH['Data_force_stat'] = train_data_path_prefix + '/Data_force/Data_force_combined.pkl'
    DATA_PATH['Data_torque_stat'] = train_data_path_prefix + '/Data_torque/Data_torque_combined.pkl'

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
                 expand_spatial=False,
                 norm_flag = False,
                 split_12 = False,
                 norm_std = 1,
                 *args,
                 **kwargs):

        self.group = group
        self.force = 'Data_force'
        self.torque = 'Data_torque'
        self.label = 'category'
        self.batch_size = batch_size
        self.expand_spatial = expand_spatial
        self.norm_flag = norm_flag
        self.norm_std = norm_std
        self.split_12 = split_12
        if norm_flag:
            self.stat_path = {}
            self.stat_path['Data_force'] = data_path['Data_force_stat']
            self.stat_path['Data_torque'] = data_path['Data_torque_stat']
        postprocess = {self.force: [(self.postprocess_images, (), {})], self.torque: [(self.postprocess_images, (), {})]}

        super(WhiskerWorld, self).__init__(
            source_dirs = [data_path["%s/%s" % (group, self.force)] , data_path["%s/%s" % (group, self.torque)] , data_path["%s/%s" % (group, self.label)]],
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

    def slice_concat_12(self, data, curr_key, new_key):
        slice0 = tf.strided_slice( data[curr_key], [0,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice1 = tf.strided_slice( data[curr_key], [1,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        slice2 = tf.strided_slice( data[curr_key], [2,0,0,0,0], [0,0,0,0,0], [3, 1, 1, 1, 1], end_mask = 31)
        data[new_key] = tf.concat([slice0, slice1, slice2], 1)

        slice0_ = tf.strided_slice( data[curr_key], [0,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        slice1_ = tf.strided_slice( data[curr_key], [1,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        slice2_ = tf.strided_slice( data[curr_key], [2,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        slice3_ = tf.strided_slice( data[curr_key], [3,0,0,0,0], [0,0,0,0,0], [4, 1, 1, 1, 1], end_mask = 31)
        data[new_key] = tf.concat([slice0_, slice1_, slice2_, slice3_], 1) 
        return data

    def spatial_slice_concat(self, data, curr_key, new_key):
        shape_now = data[curr_key].get_shape().as_list()
        slice0 = tf.slice(data[curr_key], [0, 0, 0, 0, 0], [-1, -1, 5, -1, -1])
        slice1 = tf.slice(data[curr_key], [0, 0, 5, 0, 0], [-1, -1, 6, -1, -1])
        slice2 = tf.slice(data[curr_key], [0, 0, 11, 0, 0], [-1, -1, 14, -1, -1])
        slice3 = tf.slice(data[curr_key], [0, 0, 25, 0, 0], [-1, -1, 6, -1, -1])

        pad_ten0 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
        pad_ten1 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
        pad_ten2 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])
        pad_ten3 = tf.zeros([shape_now[0], shape_now[1], 1, shape_now[3], shape_now[4]])

        data[new_key] = tf.concat([slice0, pad_ten0, pad_ten1, slice1, pad_ten2, slice2, pad_ten3, slice3], 2)
        #data[new_key] = tf.concat([slice0, slice1, slice2, slice3], 2)

        return data

    def slice_label(self, data, curr_key, new_key):
        #print(data[curr_key].get_shape().as_list())
        slice0 = tf.strided_slice( data[curr_key], [0], [0], [3], end_mask = 1)
        data[new_key] = slice0

        return data

    def slice_label_12(self, data, curr_key, new_key):
        #print(data[curr_key].get_shape().as_list())
        slice0 = tf.strided_slice( data[curr_key], [0], [0], [12], end_mask = 1)
        data[new_key] = slice0

        return data

    def normalize_data(self, data, curr_key):
        stat_dict = cPickle.load(open(self.stat_path[curr_key], 'r'))
        mean_tf = tf.constant(stat_dict['mean'], dtype = data[curr_key].dtype)
        var_tf = tf.constant(stat_dict['std'], dtype = data[curr_key].dtype)

        data[curr_key] = tf.multiply(tf.divide(tf.subtract(data[curr_key], mean_tf), var_tf), tf.constant(self.norm_std, dtype = data[curr_key].dtype))

        return data

    def init_ops(self):
        self.input_ops = super(WhiskerWorld, self).init_ops()

        # make sure batch size shapes of tensors are set
        self.input_ops = self.set_data_shapes(self.input_ops)

        for i in range(len(self.input_ops)):
            if not self.split_12:
                self.input_ops[i] = self.slice_concat(self.input_ops[i], 'Data_force', 'Data_force')
            else:
                self.input_ops[i] = self.slice_concat_12(self.input_ops[i], 'Data_force', 'Data_force')

            if self.expand_spatial:
                self.input_ops[i] = self.spatial_slice_concat(self.input_ops[i], 'Data_force', 'Data_force')

            if self.norm_flag:
                self.input_ops[i] = self.normalize_data(self.input_ops[i], 'Data_force')

            if not self.split_12:
                self.input_ops[i] = self.slice_concat(self.input_ops[i], 'Data_torque', 'Data_torque')
            else:
                self.input_ops[i] = self.slice_concat_12(self.input_ops[i], 'Data_torque', 'Data_torque')
            if self.expand_spatial:
                self.input_ops[i] = self.spatial_slice_concat(self.input_ops[i], 'Data_torque', 'Data_torque')
            if self.norm_flag:
                self.input_ops[i] = self.normalize_data(self.input_ops[i], 'Data_torque')

            if not self.split_12:
                self.input_ops[i] = self.slice_label(self.input_ops[i], 'category', 'category')
            else:
                self.input_ops[i] = self.slice_label_12(self.input_ops[i], 'category', 'category')

        return self.input_ops

    def postprocess_images(self, ims):
        def _postprocess_images(im):
            im = tf.decode_raw(im, np.float32)
            im = tf.reshape(im, [110, 31, 3, 3])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

def save_features(inputs, outputs, target, hdf5path):
    global save_num_now

    if save_num_now is None:
        save_num_now = 0

    fout = None
    if not os.path.isfile(hdf5path):
        fout = h5py.File(hdf5path, 'w')
    else:
        fout = h5py.File(hdf5path, 'a')

    all_name_list = [n.name for n in tf.get_default_graph().as_graph_def().node]

    all_name_list = filter(lambda name_now: 'fc_add' in name_now, all_name_list)
    all_name_list = filter(lambda name_now: 'validation/topn' in name_now, all_name_list)
    print(all_name_list)
    output_now = tf.get_default_graph().get_tensor_by_name('validation/topn%sfc:0' % target)
    output_shape = output_now.get_shape().as_list()
    dim_0_final = 4*2*9981
    if 'data' not in fout:
        dset = fout.create_dataset("data", [dim_0_final] + output_shape[1:], dtype='f')
    else:
        dset = fout["data"]

    dset[save_num_now:min(save_num_now + output_shape[0], dim_0_final)] = output_now[:min(output_shape[0], dim_0_final - save_num_now)]
    save_num_now = save_num_now + output_shape[0]
    return {}

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
    parser.add_argument('--loadque', default = 0, type = int, action = 'store', help = 'Special setting for load query')
    parser.add_argument('--expand', default = 0, type = int, action = 'store', help = 'Whether do the spatial padding')
    parser.add_argument('--norm', default = 0, type = int, action = 'store', help = 'Whether do the normalization, default is no')
    parser.add_argument('--split12', default = 0, type = int, action = 'store', help = 'Whether do the 12 swipes spliting, default is no')
    parser.add_argument('--norm_std', default = 1, type = float, action = 'store', help = 'Std of new input, default is 1')

    # Feature extraction related parameters
    parser.add_argument('--gen_feature', default = 0, type = int, action = 'store', help = 'Whether to generate features, default is 0, None')
    parser.add_argument('--layer_gen', default = "/cate_root/create_2/fc_add/", type = str, action = 'store', help = 'Name of layer to generate the output')
    parser.add_argument('--hdf5path', default = "/mnt/fs1/chengxuz/barrel_response/test.hdf5", type = str, action = 'store', help = 'Name of layer to generate the output')

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
                #'batch_size': BATCH_SIZE,
                'batch_size': 12,
            }
    val_data_param = {
                    'func': WhiskerWorld,
                    'data_path': DATA_PATH,
                    'group': 'val',
                    'n_threads': n_threads,
                    #'batch_size': BATCH_SIZE,
                    'batch_size': 12,
                }
    
    if args.expand==1:
        train_data_param['expand_spatial'] = True
        val_data_param['expand_spatial'] = True

    if args.norm==1:
        train_data_param['norm_flag'] = True
        val_data_param['norm_flag'] = True

        train_data_param['norm_std'] = args.norm_std
        val_data_param['norm_std'] = args.norm_std

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
            #'decay_rate': .5,
            'decay_rate': 1,
            'decay_steps': NUM_BATCHES_PER_EPOCH*10,  # exponential decay each epoch
            'staircase': True
        }

    optimizer_class = tf.train.MomentumOptimizer

    model_params = {
            'func': func_net,
            'seed': args.seed,
            'cfg_initial': cfg_initial
        }

    if args.split12==1:
        model_params['split_12'] = True
        train_data_param['split_12'] = True
        val_data_param['split_12'] = True
        train_queue_params['batch_size'] = BATCH_SIZE//12
        val_queue_params['batch_size'] = BATCH_SIZE//12
        train_queue_params['capacity'] = queue_capa//12
        val_queue_params['capacity'] = BATCH_SIZE*10//12

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

    if args.whichopt==4:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdadeltaOptimizer,
            'clip': True,
        }

    if args.whichopt==5:
        optimizer_params = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.RMSPropOptimizer,
            'clip': True,
        }

    load_query = None
    load_params = {
            'host': 'localhost',
            'port': args.nport,
            'dbname': 'whisker_net',
            'collname': 'catenet',
            'exp_id': exp_id,
            'do_restore': True,
            'query': load_query 
    }

    if args.loadque==1:
        load_query = {'saved_filters': True, 'step': 70000}
        load_params = {
                'host': 'localhost',
                'port': args.nport,
                'dbname': 'whisker_net',
                'collname': 'catenet',
                'exp_id': 'catenet_adag_flv_slac_2',
                'do_restore': True,
                'query': load_query 
        }
        #print(load_query)

    save_params = {
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
        }

    train_params = {
            'validate_first': False,
            'data_params': train_data_param,
            'queue_params': train_queue_params,
            'thres_loss': 1000000000,
            'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
        }

    loss_params = {
            'targets': val_target,
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_func,
        }

    validation_params = {
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
        }

    if args.gen_feature==1:
        train_params['validate_first'] = True
        train_params['num_steps'] = 1
        
        validation_params['topn']['targets'] = {
                'func': save_features,
                'target': args.layer_gen,
                'hdf5path': args.hdf5path
            }


    params = {
        'save_params': save_params,

        'load_params': load_params,

        'model_params': model_params,

        'train_params': train_params,

        'loss_params': loss_params,

        'learning_rate_params': learning_rate_params,

        'optimizer_params': optimizer_params,

        'log_device_placement': False,  # if variable placement has to be logged
        'validation_params': validation_params,
    }
    #base.get_params()
    base.train_from_params(**params)

if __name__ == '__main__':
    main()
