import h5py
import numpy as np
import os, sys

import argparse
import cPickle
import time

from scipy.stats import pearsonr

from cmp_asgn import get_file_list

ALL_DATA = {}

def load_or_fetch(data_path, all_data = ALL_DATA):
    if not data_path in all_data:
        print("Loading %s" % data_path)
        sys.stdout.flush()
        fin = h5py.File(data_path, 'r')
        RDM_arr = np.asarray(fin['RDM'])
        all_data[data_path] = RDM_arr.reshape(RDM_arr.size)

    return all_data[data_path]

def main():
    parser = argparse.ArgumentParser(description='The script to compute partial of distance matrix')
    parser.add_argument('--savefolder', default = '/mnt/fs0/chengxuz/Data/nd_response/partial_dismat', type = str, action = 'store', help = 'Folder to save the result')
    parser.add_argument('--saveprefix', default = 'Part', type = str, action = 'store', help = 'Prefix to save the result')
    parser.add_argument('--savesuffix', default = '.pkl', type = str, action = 'store', help = 'Prefix to save the result')
    parser.add_argument('--xstart', default = 0, type = int, action = 'store', help = 'Start index for x')
    parser.add_argument('--xlen', default = 1, type = int, action = 'store', help = 'Length of index list for x')
    parser.add_argument('--ystart', default = 0, type = int, action = 'store', help = 'Start index for y')
    parser.add_argument('--ylen', default = 1, type = int, action = 'store', help = 'Length of index list for y')
    parser.add_argument('--filelen', default = None, type = int, action = 'store', help = 'Length of the whole file list')

    args    = parser.parse_args()

    os.system('mkdir -p %s' % args.savefolder)

    file_list = get_file_list()
    if args.filelen is None:
        args.filelen = len(file_list)
    else:
        assert args.filelen > len(file_list), "Must be smaller than whole length"
        file_list = file_list[:args.filelen]

    res_dict = {}
    x_indx_list = range(args.xstart, min(args.xstart + args.xlen, args.filelen))
    y_indx_list = range(args.ystart, min(args.ystart + args.ylen, args.filelen))
    x_name_list = [file_list[v] for v in x_indx_list]
    y_name_list = [file_list[v] for v in y_indx_list]
    part_dismat = np.zeros([args.xlen, args.ylen])

    for nx_indx, x_indx in enumerate(x_indx_list):
        x_arr_now = load_or_fetch(file_list[x_indx])

        for ny_indx, y_indx in enumerate(y_indx_list):
            y_arr_now = load_or_fetch(file_list[y_indx])

            part_dismat[nx_indx, ny_indx] = pearsonr(x_arr_now, y_arr_now)[0]

            print(nx_indx, ny_indx)
            sys.stdout.flush()

    res_dict['x_name_list'] = x_name_list
    res_dict['y_name_list'] = y_name_list
    res_dict['part_dismat'] = 1 - part_dismat

    tmp_path = os.path.join(args.savefolder, '%s_%i_%i_%i_%i_len%i_%s' % (args.saveprefix, args.xstart, args.xlen, args.ystart, args.ylen, args.filelen, args.savesuffix))
    cPickle.dump(res_dict, open(tmp_path, 'w'))
    pass

if __name__ == '__main__':
    main()
