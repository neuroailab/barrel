import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import time

def main():
    parser = argparse.ArgumentParser(description='The script to compute category level RDM')

    parser.add_argument('--hdf5path', default = '/data2/chengxuz/nd_response/resp_temp_spa_s0_2.hdf5', type = str, action = 'store', help = 'Path to the original hdf5 file storing all the information')
    parser.add_argument('--key', default = 'conv4_0', type = str, action = 'store', help = 'which key in the hdf5 file to use')
    parser.add_argument('--savefolder', default = '/mnt/fs0/chengxuz/Data/nd_response/temp_spa_responses_s0', type = str, action = 'store', help = 'Path to store the extracted responses')
    parser.add_argument('--batchsize', default = 100, type = int, action = 'store', help = 'Number of reponses per batch')

    args    = parser.parse_args()

    fin = h5py.File(args.hdf5path, 'r')
    shape_now = fin[args.key].shape
    len_file = shape_now[0]

    os.system('mkdir -p %s' % args.savefolder)
    save_path = os.path.join(args.savefolder, '%s.hdf5' % args.key)
    fout = h5py.File(save_path, 'w')

    dataset = fout.create_dataset(args.key, shape_now, dtype = 'f')

    for indx_batch in xrange(0, len_file, args.batchsize):
        indx_end = min(indx_batch + args.batchsize, len_file)

        start_time = time.time()
        dataset[indx_batch:indx_end] = fin[args.key][indx_batch:indx_end]
        end_time = time.time()

        print('Batch %i take time %f' % (indx_batch//args.batchsize, end_time - start_time))

    fin.close()
    fout.close()

if __name__ == '__main__':
    main()
