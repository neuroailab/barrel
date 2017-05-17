import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import time

def main():
    parser = argparse.ArgumentParser(description='The script to compute category level RDM')

    parser.add_argument('--hdf5path', default = '/data2/chengxuz/nd_response/spatemp_responses.hdf5', type = str, action = 'store', help = 'Path to the original hdf5 file storing all the information')
    parser.add_argument('--key', default = 'conv1_0', type = str, action = 'store', help = 'which key in the hdf5 file to use')
    parser.add_argument('--savepath', default = '/mnt/fs0/chengxuz/Data/nd_response/RDM_conv1_0.pkl', type = str, action = 'store', help = 'Path to store the final RDM')
    parser.add_argument('--numcat', default = 117, type = int, action = 'store', help = 'Number of categories')
    parser.add_argument('--labelkey', default = 'label', type = str, action = 'store', help = 'Key used for label')
    parser.add_argument('--labelfile', default = None, type = str, action = 'store', help = 'hdf5 file used for label')
    parser.add_argument('--writeway', default = 0, type = int, action = 'store', help = '0 means cPickle, 1 means hdf5')
    parser.add_argument('--keyaslist', default = 0, type = int, action = 'store', help = '0 means not list, 1 means as list')
    parser.add_argument('--keystart', default = 0, type = int, action = 'store', help = 'Start key')
    parser.add_argument('--keyend', default = 0, type = int, action = 'store', help = 'End key')
    
    #parser.add_argument('--hdf5list', default = [], type = str, action = 'append', help = 'Path to the list of hdf5 file storing all the information')

    args    = parser.parse_args()

    print(args.key)

    fin = h5py.File(args.hdf5path, 'r')

    #fin_list = []

    key_tmp = args.key
    if args.keyaslist==1:
        key_tmp = args.key % args.keystart

    num_example = fin[key_tmp].shape[0]
    fea_len = fin[key_tmp].size/num_example
    all_sum = np.zeros([args.numcat, fea_len])
    all_num = np.zeros(args.numcat)

    if args.labelfile is None:
        label_fin = fin
    else:
        label_fin = h5py.File(args.labelfile, 'r')
    all_label = np.asarray(label_fin[args.labelkey])


    start_time = time.time()

    if args.keyaslist==0:
        key_list = [args.key]
    else:
        key_list = [args.key % v for v in xrange(args.keystart, args.keyend)]

    for indx_which in xrange(num_example):
        which_cat = int(all_label[indx_which])


        for key_now in key_list:
            tmp_arr = np.asarray(fin[key_now][indx_which])
            all_sum[which_cat] = all_sum[which_cat] + tmp_arr.reshape(tmp_arr.size)
            all_num[which_cat] = all_num[which_cat] + 1

        if indx_which%2000==0:
            print('Now num %i' % indx_which)

    all_mean = all_sum/all_num[:, None]

    load_time = time.time()
    print('Loading time %f' % (load_time - start_time))

    dis_matrix = 1 - np.corrcoef(all_mean)

    dis_time = time.time()
    print('Dis_mat time %f' % (dis_time - load_time))

    if args.writeway==0:
        cPickle.dump(dis_matrix, open(args.savepath, 'w'))
    else:
        fout = h5py.File(args.savepath, 'w')
        dataset = fout.create_dataset('RDM', dis_matrix.shape, dtype='f')
        dataset[...] = dis_matrix
        fout.close()

    dump_time = time.time()
    print('Dump time %f' % (dump_time - dis_time))


    fin.close()
    if not args.labelfile is None:
        label_fin.close()

if __name__ == '__main__':
    main()
