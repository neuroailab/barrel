import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import time

def main():
    parser = argparse.ArgumentParser(description='The script to compute category level RDM')

    parser.add_argument('--hdf5path', default = '/data/chengxuz/nd_response/resp_spa_temp_2.hdf5', type = str, action = 'store', help = 'Path to the original hdf5 file storing all the information')
    parser.add_argument('--key', default = 'conv1_0', type = str, action = 'store', help = 'which key in the hdf5 file to use')
    parser.add_argument('--savefolder', default = '/data/chengxuz/nd_response/spa_temp_resps', type = str, action = 'store', help = 'Path to store the extracted responses')
    parser.add_argument('--savename', default = 'conv1.hdf5', type = str, action = 'store', help = 'Path to store the extracted responses')
    parser.add_argument('--batchsize', default = 100, type = int, action = 'store', help = 'Number of reponses per batch')
    parser.add_argument('--numcat', default = 9981, type = int, action = 'store', help = 'Number of categories')
    parser.add_argument('--labelkey', default = 'objid', type = str, action = 'store', help = 'Key used for label')
    parser.add_argument('--labelfile', default = '/mnt/fs0/chengxuz/Data/nd_response/responses_otherlabels_2.hdf5', type = str, action = 'store', help = 'hdf5 file used for label')
    parser.add_argument('--check', default = 0, type = int, action = 'store', help = '1 means do it anyway')

    args    = parser.parse_args()

    print('Key now %s' % args.key)

    os.system('mkdir -p %s' % args.savefolder)

    fin = h5py.File(args.hdf5path, 'r')

    key_tmp = args.key

    num_example = fin[key_tmp].shape[0]
    fea_len = fin[key_tmp].size/num_example
    all_sum = np.zeros([args.numcat, fea_len])
    all_num = np.zeros(args.numcat)

    label_fin = h5py.File(args.labelfile, 'r')
    all_label = np.asarray(label_fin[args.labelkey])

    save_path = os.path.join(args.savefolder, args.savename)
    fout = h5py.File(save_path, 'a')

    if args.key in fout:
        if args.check==0:
            return
        dataset  = fout[args.key]
    else:
        dataset = fout.create_dataset(args.key, [args.numcat, fea_len], dtype = 'f')

    for indx_batch in xrange(0, num_example, args.batchsize):
        indx_end = min(indx_batch + args.batchsize, num_example)

        #start_time = time.time()
        curr_array = np.asarray(fin[args.key][indx_batch:indx_end])
        curr_array = curr_array.reshape([curr_array.shape[0], -1])
        for indx_tmp in xrange(indx_batch, indx_end):
            which_cat = int(all_label[indx_tmp])

            all_sum[which_cat] = all_sum[which_cat] + curr_array[indx_tmp - indx_batch]
            all_num[which_cat] = all_num[which_cat] + 1

        #end_time = time.time()

        #print('Batch %i take time %f' % (indx_batch//args.batchsize, end_time - start_time))

    all_mean = all_sum/all_num[:, None]
    dataset[...] = all_mean

    fin.close()
    fout.close()
    label_fin.close()

if __name__ == '__main__':
    main()
