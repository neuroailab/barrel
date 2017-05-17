import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import time

from scipy.stats import pearsonr

from cmp_asgn import get_file_list

def main():
    parser = argparse.ArgumentParser(description='The script to combine the partial dismat')
    parser.add_argument('--loadfolder', default = '/mnt/fs0/chengxuz/Data/nd_response/partial_dismat', type = str, action = 'store', help = 'Folder to load the result from')
    parser.add_argument('--loadprefix', default = 'Part', type = str, action = 'store', help = 'Prefix to load the result')
    parser.add_argument('--loadsuffix', default = '.pkl', type = str, action = 'store', help = 'Prefix to load the result')
    parser.add_argument('--xlen', default = 10, type = int, action = 'store', help = 'Length of index list for x')
    parser.add_argument('--ylen', default = 10, type = int, action = 'store', help = 'Length of index list for y')
    parser.add_argument('--savepath', default = '/mnt/fs0/chengxuz/Data/nd_response/all_dismat.pkl', type = str, action = 'store', help = 'Path to save the result')

    args    = parser.parse_args()

    file_list = get_file_list()
    file_len = len(file_list)

    non_num = 0

    for xstart in xrange(0, file_len, args.xlen):
        for ystart in xrange(xstart, file_len, args.ylen):
            path_now = os.path.join(args.loadfolder, '%s_%i_%i_%i_%i_len%i_%s' % (args.loadprefix, xstart, args.xlen, ystart, args.ylen, file_len, args.loadsuffix))

            if not os.path.exists(path_now):
                print(path_now)
                non_num = non_num + 1

    print(non_num)

    big_dismat = np.zeros([file_len, file_len])
    if non_num==0:
        for xstart in xrange(0, file_len, args.xlen):
            for ystart in xrange(xstart, file_len, args.ylen):
                path_now = os.path.join(args.loadfolder, '%s_%i_%i_%i_%i_len%i_%s' % (args.loadprefix, xstart, args.xlen, ystart, args.ylen, file_len, args.loadsuffix))

                now_dict = cPickle.load(open(path_now, 'r'))
                part_dismat = now_dict['part_dismat']

                for xindx in xrange(min(args.xlen, file_len - xstart)):
                    for yindx in xrange(min(args.ylen, file_len - ystart)):
                        big_dismat[xindx + xstart, yindx + ystart] = part_dismat[xindx, yindx]

                for yindx in xrange(min(args.ylen, file_len - ystart)):
                    for xindx in xrange(min(args.xlen, file_len - xstart)):
                        big_dismat[yindx + ystart, xindx + xstart] = part_dismat[xindx, yindx]

    res_dict = {'file_list': file_list, 'dismat': big_dismat}
    cPickle.dump(res_dict, open(args.savepath, 'w'))

if __name__ == '__main__':
    main()
