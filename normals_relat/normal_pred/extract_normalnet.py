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

#os.environ["CUDA_VISIBLE_DEVICES"]="2"

host = os.uname()[1]

DATA_PATH = {}
if host == 'freud':  # freud
    DATA_PATH['train'] = '/media/data/one_world_dataset/randomperm.hdf5'
    DATA_PATH['val'] = '/media/data/one_world_dataset/randomperm_test1.hdf5'

elif host.startswith('node') or host == 'openmind7':  # OpenMind
    DATA_PATH['train'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm.hdf5'
    DATA_PATH['val'] = '/om/user/chengxuz/Data/one_world_dataset/randomperm_test1.hdf5'
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

    #N_TRAIN = 2048000 - 102400
    #N_VAL = 102400 
    N_TRAIN = 2048000
    #N_VAL = 128000
    N_VAL = 256

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
        '''
        if self.group=='train':
            subslice = range(self.N_TRAIN)
        else:
            subslice = range(self.N_TRAIN, self.N_TRAIN + self.N_VAL)
        '''
        super(Threedworld, self).__init__(
            data_path[group],
            [self.images, self.labels],
            batch_size=batch_size,
            postprocess={self.images: self.postproc, self.labels: self.postproc},
            pad=True,
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
#BATCH_SIZE = 192
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


def main(args):
    #cfg_initial = postprocess_config(json.load(open(cfgfile)))
    if args.gpu>-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg_initial = preprocess_config(json.load(open(args.pathconfig)))
    exp_id  = args.expId
    cache_dir = os.path.join(args.cacheDirPrefix, '.tfutils', 'localhost:'+ str(args.nport), 'normalnet-test', 'normalnet', exp_id)

    """
    This is a test illustrating how to perform feature extraction using
    tfutils.base.test_from_params.
    The basic idea is to specify a validation target that is simply the actual output of
    the model at some layer. (See the "get_extraction_target" function above as well.)
    This test assumes that test_train has run first.

    After the test is run, the results of the feature extraction are saved in the Grid
    File System associated with the mongo database, with one file per batch of feature
    results.  See how the features are accessed by reading the test code below.
    """
    # set up parameters
    params = {}
    params['model_params'] = {
            'func': normal_encoder_asymmetric_with_bypass.normalnet_tfutils,
            'seed': args.seed,
            'cfg_initial': cfg_initial
            }
    params['load_params'] = {'host': 'localhost',
                             'port': 22334,
                             'dbname': 'normalnet-test',
                             'collname': 'normalnet',
                             'do_restore': True,
                             'exp_id': exp_id}
    #params['save_params'] = {'exp_id': 'validation1',
    params['save_params'] = {'exp_id': 'validation1',
                             'save_intermediate_freq': 1,
                             'save_to_gfs': ['features']}

    targdict = {'func': get_extraction_target,
                'to_extract': {'features': 'validation/valid1/dec7/conv:0'},
                }
    targdict.update(base.DEFAULT_LOSS_PARAMS)
    params['validation_params'] = {'valid1': {'data_params': {'func': Threedworld,
                                                              'data_path': DATA_PATH,
                                                              'batch_size': 1,
                                                              'crop_size': IMAGE_SIZE_CROP,
                                                              'group': 'val'},
                                              'queue_params': {'queue_type': 'fifo',
                                                               'batch_size': 100,
                                                               'n_threads': 4},
                                              'targets': targdict,
                                              'num_steps': 2,
                                              'online_agg_func': utils.reduce_mean_dict}}
    '''
    params['loss_params'] = {
            'targets': 'labels',
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_ave_l2
        }
    '''

    # actually run the feature extraction
    #base.test_from_params(**params)

    # check that things are as expected.
    conn = pm.MongoClient(host='localhost',
                          port=22334)
    coll = conn['normalnet-test']['normalnet'+'.files']
    #assert coll.find({'exp_id': 'validation1'}).count() == 11
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
    idval = r['validation_results']['valid1']['intermediate_steps'][0]
    fn = coll.find({'item_for': idval})[0]['filename']
    fs = gridfs.GridFS(coll.database, 'normalnet')
    fh = fs.get_last_version(fn)
    saved_data = cPickle.loads(fh.read())
    fh.close()
    features = saved_data['validation_results']['valid1']['features']
    print(features.shape)
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

    args    = parser.parse_args()

    main(args)
