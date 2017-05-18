import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import time

from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser(description='The script to compute partial of RDM')
    parser.add_argument('--savefolder', default = '/mnt/fs0/chengxuz/Data/nd_response/partial_RDM', type = str, action = 'store', help = 'Folder to save the result')
    parser.add_argument('--saveprefix', default = 'Part', type = str, action = 'store', help = 'Prefix to save the result')
    parser.add_argument('--savesuffix', default = '.pkl', type = str, action = 'store', help = 'Prefix to save the result')
    parser.add_argument('--xstart', default = 0, type = int, action = 'store', help = 'Start index for x')
    parser.add_argument('--xlen', default = 1000, type = int, action = 'store', help = 'Length of index list for x')
    parser.add_argument('--ystart', default = 0, type = int, action = 'store', help = 'Start index for y')
    parser.add_argument('--ylen', default = 1000, type = int, action = 'store', help = 'Length of index list for y')
    parser.add_argument('--filelen', default = 9981, type = int, action = 'store', help = 'Length of the whole file list')

    parser.add_argument('--hdf5path', default = '/mnt/fs0/chengxuz/Data/nd_response/spa_temp_resps/conv1.hdf5', type = str, action = 'store', help = 'Path to get the responses')
    parser.add_argument('--keypat', default = 'conv1_%i', type = str, action = 'store', help = 'Patter for keys to compute')
    parser.add_argument('--keystart', default = 0, type = int, action = 'store', help = 'Start index for key')
    parser.add_argument('--keyend', default = 22, type = int, action = 'store', help = 'End of index for key')

    args    = parser.parse_args()

    os.system('mkdir -p %s' % args.savefolder)

    fin = h5py.File(args.hdf5path, 'r')

    x_indx_list = range(args.xstart, min(args.xstart + args.xlen, args.filelen))
    y_indx_list = range(args.ystart, min(args.ystart + args.ylen, args.filelen))
    x_arr = np.concatenate([fin[args.keypat % v][x_indx_list] for v in xrange(args.keystart, args.keyend)], 1)
    y_arr = np.concatenate([fin[args.keypat % v][y_indx_list] for v in xrange(args.keystart, args.keyend)], 1)

    part_RDM = np.zeros([args.xlen, args.ylen])

    for nx_indx, x_indx in enumerate(x_indx_list):
        for ny_indx, y_indx in enumerate(y_indx_list):

            part_RDM[nx_indx, ny_indx] = pearsonr(x_arr[nx_indx], y_arr[ny_indx])[0]

        if nx_indx%20==0:
            print(nx_indx)
            sys.stdout.flush()

    tmp_path = os.path.join(args.savefolder, '%s_%i_%i_%i_%i_len%i_%s' % (args.saveprefix, args.xstart, args.xlen, args.ystart, args.ylen, args.filelen, args.savesuffix))
    cPickle.dump(part_RDM, open(tmp_path, 'w'))
    pass

if __name__ == '__main__':
    main()
