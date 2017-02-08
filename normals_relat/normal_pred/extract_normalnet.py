"""
Script used for extracting the normalnet using tfutils
"""

from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf

import normal_encoder_asymmetric_with_bypass

from tfutils import base, data, model, optimizer, utils
import cPickle

import pymongo as pm
import gridfs

import json
import copy
import argparse

import train_normalnet_hdf5
import train_normalnet



def loss_ave_l2(output, labels):
    loss = tf.nn.l2_loss(output - labels) / NORM_NUM
    return loss

def rep_loss(inputs, outputs, target):
    loss = tf.nn.l2_loss(outputs - inputs[target]) / NORM_NUM
    #output_ary = outputs.eval()
    print('Temp')
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

def get_extraction_target(inputs, outputs, to_extract, **loss_params):
    """
    Example validation target function to use to provide targets for extracting features.
    This function also adds a standard "loss" target which you may or not may not want

    The to_extract argument must be a dictionary of the form
          {name_for_saving: name_of_actual_tensor, ...}
    where the "name_for_saving" is a human-friendly name you want to save extracted
    features under, and name_of_actual_tensor is a name of the tensor in the tensorflow
    graph outputing the features desired to be extracted.  To figure out what the names
    of the tensors you want to extract are "to_extract" argument,  uncomment the
    commented-out lines, which will print a list of all available tensor names.
    """

    # names = [[x.name for x in op.values()] for op in tf.get_default_graph().get_operations()]
    # print("NAMES are: ", names)

    loss_params['loss_per_case_func'] = loss_ave_l2
    loss_params['loss_per_case_func_params'] = {}

    targets = {k: tf.get_default_graph().get_tensor_by_name(v) for k, v in to_extract.items()}
    targets['loss'] = utils.get_loss(inputs, outputs, **loss_params)
    return targets

def get_current_predicted_future_action(inputs, outputs, num_to_save = 1, **loss_params):

    images = inputs['images'][:num_to_save]
    images = tf.cast(images, tf.uint8)

    normals = inputs['normals'][:num_to_save]
    normals = tf.cast(normals, tf.uint8)

    loss_params['loss_per_case_func'] = loss_ave_l2
    loss_params['loss_per_case_func_params'] = {}
    loss = utils.get_loss(inputs, outputs, **loss_params)

    retval = {'images' : images, 'normals' : normals, \
              'val_loss': loss}
    return retval


def main(args):
    #cfg_initial = postprocess_config(json.load(open(cfgfile)))
    if args.gpu>-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg_initial = preprocess_config(json.load(open(args.pathconfig)))
    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)

    host = os.uname()[1]
    DATA_PATH = {}
    if args.hdf5ortfc==0:
        Threedworld = train_normalnet_hdf5.Threedworld

        if host == 'freud':  # freud
            DATA_PATH['train'] = '/media/data/one_world_dataset/randomperm.hdf5'
            DATA_PATH['val'] = '/media/data/one_world_dataset/randomperm_test1.hdf5'

        elif host.startswith('node') or host == 'openmind7':  # OpenMind
            DATA_PATH['train'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm.hdf5'
            DATA_PATH['val'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm_test1.hdf5'
    else:
        Threedworld = train_normalnet.Threedworld

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

    BATCH_SIZE = 128
    NUM_BATCHES_PER_EPOCH = Threedworld.N_TRAIN // BATCH_SIZE
    IMAGE_SIZE_CROP = 224
    NUM_CHANNELS = 3
    NORM_NUM = (IMAGE_SIZE_CROP**2) * NUM_CHANNELS * BATCH_SIZE

    # set up parameters
    params = {}
    params['model_params'] = {
            'func': normal_encoder_asymmetric_with_bypass.normalnet_tfutils,
            'seed': args.seed,
            'cfg_initial': cfg_initial
            }
    params['load_params'] = {'host': 'localhost',
                             'port': args.nport,
                             'dbname': 'normalnet-test',
                             'collname': 'normalnet',
                             'do_restore': True,
                             'exp_id': exp_id}
    #params['save_params'] = {'exp_id': 'validation1',
    params['save_params'] = {'exp_id': 'validation1',
                             'save_intermediate_freq': 1,
                             'save_to_gfs': ['features']}

    '''
    targdict = {'func': get_extraction_target,
                'to_extract': {'features': 'validation/valid1/dec7/conv:0'},
                }
    '''
    targdict = {'func': get_current_predicted_future_action,
                'targets' : [],
                'num_to_save' : 10
                }
    targdict.update(base.DEFAULT_LOSS_PARAMS)
    params['validation_params'] = {'valid1': {
            'data_params': {
                'func': Threedworld,
                'data_path': DATA_PATH,
                'group': 'train',
                'crop_size': IMAGE_SIZE_CROP,
                'batch_size': BATCH_SIZE,
                'n_threads' : 1
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': BATCH_SIZE,
                'seed': 0,
                'capacity': BATCH_SIZE*20,
            },
            'targets': targdict,
            'num_steps': 8,
            'online_agg_func': utils.reduce_mean_dict}}
    '''
    params['loss_params'] = {
            'targets': 'labels',
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_ave_l2
        }
    '''

    # actually run the feature extraction
    base.test_from_params(**params)

    # check that things are as expected.
    conn = pm.MongoClient(host='localhost',
                          port=args.nport)
    coll = conn['normalnet-test']['normalnet'+'.files']

    print(coll.find({'exp_id': 'validation1'}).count())

    # ... load the containing the final "aggregate" result after all features have been extracted
    q = {'exp_id': 'validation1', 'validation_results.valid1.intermediate_steps': {'$exists': True}}
    #assert coll.find(q).count() == 1
    print(coll.find(q).count())
    r = coll.find(q)[0]
    # ... check that the record is well-formed
    #asserts_for_record(r, params, train=False)

    # ... check that the correct "intermediate results" (the actual features extracted) records exist
    # and are correctly referenced.
    q1 = {'exp_id': 'validation1', 'validation_results.valid1.intermediate_steps': {'$exists': False}}
    ids = coll.find(q1).distinct('_id')
    #assert r['validation_results']['valid1']['intermediate_steps'] == ids

    # ... actually load feature batch 3
    #idval = r['validation_results']['valid1']['intermediate_steps'][0]
    idval = r['validation_results']['valid1']['intermediate_steps'][2]
    fn = coll.find({'item_for': idval})[0]['filename']
    fs = gridfs.GridFS(coll.database, 'normalnet')
    fh = fs.get_last_version(fn)
    saved_data = cPickle.loads(fh.read())
    fh.close()
    features = saved_data['validation_results']['valid1']['features']
    print(features.shape)
    #cPickle.dump(saved_data, open('save_features.pkl', 'wb'))
    #cPickle.dump(saved_data, open('save_features_3.pkl', 'wb'))
    cPickle.dump(saved_data, open('save_features_4.pkl', 'wb'))
    #assert features.shape == (100, 128)
    #assert features.dtype == np.float32

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
    parser.add_argument('--hdf5ortfc', default = 1, type = int, action = 'store', help = 'default is 0, 0 for hdf5, 1 for tfrecords')

    args    = parser.parse_args()

    main(args)
