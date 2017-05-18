import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import time

from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser(description='The script to combine the partial dismat')
    parser.add_argument('--loadfolder', default = '/mnt/fs0/chengxuz/Data/nd_response/partial_RDM', type = str, action = 'store', help = 'Folder to load the result from')
    parser.add_argument('--loadprefix', default = 'Part', type = str, action = 'store', help = 'Prefix to load the result')
    parser.add_argument('--loadsuffix', default = 'conv6', type = str, action = 'store', help = 'Prefix to load the result')
    parser.add_argument('--xlen', default = 1000, type = int, action = 'store', help = 'Length of index list for x')
    parser.add_argument('--ylen', default = 1000, type = int, action = 'store', help = 'Length of index list for y')
    parser.add_argument('--savepath', default = '/mnt/fs0/chengxuz/Data/nd_response/spa_temp_RDMs_obj/RDM_conv6_concat0_22.hdf5', type = str, action = 'store', help = 'Path to save the result')
    parser.add_argument('--filelen', default = 9981, type = int, action = 'store', help = 'Whole length')

    args    = parser.parse_args()

    file_len = args.filelen
    non_num = 0

    for xstart in xrange(0, file_len, args.xlen):
        for ystart in xrange(xstart, file_len, args.ylen):
            path_now = os.path.join(args.loadfolder, '%s_%i_%i_%i_%i_len%i_%s' % (args.loadprefix, xstart, args.xlen, ystart, args.ylen, file_len, args.loadsuffix))

            if not os.path.exists(path_now):
                print(path_now)
                non_num = non_num + 1

    print(non_num)

    if non_num>0:
        return

    big_dismat = np.zeros([file_len, file_len])

    for xstart in xrange(0, file_len, args.xlen):
        for ystart in xrange(xstart, file_len, args.ylen):
            path_now = os.path.join(args.loadfolder, '%s_%i_%i_%i_%i_len%i_%s' % (args.loadprefix, xstart, args.xlen, ystart, args.ylen, file_len, args.loadsuffix))

            part_dismat = cPickle.load(open(path_now, 'r'))

            for xindx in xrange(min(args.xlen, file_len - xstart)):
                for yindx in xrange(min(args.ylen, file_len - ystart)):
                    big_dismat[xindx + xstart, yindx + ystart] = part_dismat[xindx, yindx]

            for yindx in xrange(min(args.ylen, file_len - ystart)):
                for xindx in xrange(min(args.xlen, file_len - xstart)):
                    big_dismat[yindx + ystart, xindx + xstart] = part_dismat[xindx, yindx]

    fout = h5py.File(args.savepath, 'w')
    dataset = fout.create_dataset('RDM', big_dismat.shape, dtype='f')
    dataset[...] = big_dismat
    fout.close()

if __name__ == '__main__':
    main()
