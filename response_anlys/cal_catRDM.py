import h5py
import numpy as np
import os, sys

import argparse
import cPickle

def main():
    parser = argparse.ArgumentParser(description='The script to compute category level RDM')

    parser.add_argument('--hdf5path', default = '/data2/chengxuz/nd_response/spatemp_responses.hdf5', type = str, action = 'store', help = 'Path to the original hdf5 file storing all the information')
    parser.add_argument('--key', default = 'conv1_0', type = str, action = 'store', help = 'which key in the hdf5 file to use')
    parser.add_argument('--savepath', default = '/mnt/fs0/chengxuz/Data/nd_response/RDM_conv1_0.pkl', type = str, action = 'store', help = 'Path to store the final RDM')
    parser.add_argument('--numcat', default = 117, type = int, action = 'store', help = 'Number of categories')

    args    = parser.parse_args()

    print(args.key)

    fin = h5py.File(args.hdf5path, 'r')

    num_example = fin[args.key].shape[0]
    fea_len = fin[args.key].size/num_example
    all_sum = np.zeros([args.numcat, fea_len])
    all_num = np.zeros(args.numcat)

    all_label = np.asarray(fin['label'])

    for indx_which in xrange(num_example):
        which_cat = int(all_label[indx_which])
        tmp_arr = np.asarray(fin[args.key][indx_which])
        all_sum[which_cat] = all_sum[which_cat] + tmp_arr.reshape(tmp_arr.size)
        all_num[which_cat] = all_num[which_cat] + 1

        if indx_which%2000==0:
            print('Now num %i' % indx_which)

    all_mean = all_sum/all_num[:, None]

    dis_matrix = 1 - np.corrcoef(all_mean)

    cPickle.dump(dis_matrix, open(args.savepath, 'w'))

if __name__ == '__main__':
    main()
