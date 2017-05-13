import h5py
import numpy as np
import os, sys

import argparse
import tabular as tb
import cPickle

def main():
    parser = argparse.ArgumentParser(description='The script to generate the meta file for validation dataset')

    parser.add_argument('--hdf5path', default = '/mnt/fs0/chengxuz/Data/nd_response/responses_otherlabels.hdf5', type = str, action = 'store', help = 'Path to the original hdf5 file storing all the information')
    parser.add_argument('--metapath', default = '/mnt/fs0/chengxuz/Data/nd_response/meta.pkl', type = str, action = 'store', help = 'Path to store the final meta')

    args    = parser.parse_args()

    label_list = [u'label', u'orn', u'position', u'scale', u'speed']
    index_lists = [[0], [0,1,2,3], [0,1,2], [3], [0,1,2]]

    name_list = []
    column_list = []
    # fill in columns, names
    fin = h5py.File(args.hdf5path, 'r')
    for label, index_list in zip(label_list, index_lists):

        arr_now = np.asarray(fin[label])
        for nindx, index in enumerate(index_list):

            if len(arr_now.shape)>1:
                column_list.append(arr_now[:, index])
            else:
                column_list.append(arr_now[:])
            if len(index_list)>1:
                name_list.append("%s_%i" % (label, nindx))
            else:
                name_list.append(label)

    #print(column_lists)
    #print(name_list)

    meta_now = {'columns': column_list, 'names': name_list}

    cPickle.dump(meta_now, open(args.metapath, 'w'))


if __name__ == '__main__':
    main()
