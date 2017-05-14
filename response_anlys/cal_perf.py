import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import tabular as tb
import dldata.metrics.utils as dlutils

def main():
    parser = argparse.ArgumentParser(description='The script to compute various performance')

    parser.add_argument('--metapath', default = '/mnt/fs0/chengxuz/Data/nd_response/meta.pkl', type = str, action = 'store', help = 'Path to the meta information')
    parser.add_argument('--savepath', default = '/mnt/fs0/chengxuz/Data/nd_response/perf_fc_add.pkl', type = str, action = 'store', help = 'Path to store the final RDM')

    args    = parser.parse_args()

    meta_dict = cPickle.load(open(args.metapath, 'r'))
    meta_now = tb.tabarray(**meta_dict)

    fin = h5py.File('/data2/chengxuz/nd_response/spatemp_responses.hdf5')
    #F_arr = np.asarray(fin['fc7_0'])
    #F_arr = np.asarray(fin['fc6_0'])
    F_arr = np.asarray(fin['conv5_0'])
    F_arr = F_arr.reshape([F_arr.shape[0], F_arr.size/F_arr.shape[0]])
    F_arr = F_arr[:, np.random.RandomState(0).permutation(F_arr.shape[1])[:1024]]
    #F_arr = np.asarray(fin['fc_add_0'])
    print(F_arr.shape)
    eval_config = {'npc_train': 80, 
		   'npc_test': 200, 
		   'num_splits': 1,
		   'metric_screen': 'regression',
		   #'metric_kwargs': {'model_type': 'linear_model.LinearRegression', 
		   'metric_kwargs': {'model_type': 'linear_model.Ridge', 
                       'model_kwargs': {'alpha': 1}},
		   #'metric_screen': 'classifier',
		   #'metric_kwargs': {'model_type': 'svm.LinearSVC', 
                   #                  'model_kwargs': {'C': 5e-3}},
                   #'labelfunc': 'scale',
                   'labelfunc': lambda x: (np.asarray([x['speed_0'], x['speed_1'], x['speed_2']]).transpose(), None),
                   #'labelfunc': 'label',
		   'npc_validate': 0,
		   'split_by': 'label',
		   'test_q': None, 
		   'train_q': None}
    result = dlutils.compute_metric_base(F_arr,
                                         meta_now, eval_config)
    print(result['multi_rsquared_array_loss'])
    #cPickle.dump(result, open(args.savepath, 'w'))

if __name__ == '__main__':
    main()
