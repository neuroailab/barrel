import os
import numpy as np
#import tensorflow as tf
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compute the statistics of the input in order to normalize')
    parser.add_argument('--tflen', default = 1, type = int, action = 'store', help = 'Number of tfrecs for computing')
    parser.add_argument('--tfstart', default = 0, type = int, action = 'store', help = 'Start index of tfrecs for computing')
    parser.add_argument('--tfdir', default = '/scratch/users/chengxuz/barrel/barrel_relat_files/dataset/tfrecords/Data_force', type = str, action = 'store', help = 'Directory holding the tfrec files')

    args    = parser.parse_args()

    shape_tfr = [110, 31, 3, 3]
    obj_len = 25 # Number of objs in one tfrecord file
    name_pat = 'Data%i_%i.tfrecords'

    for tf_indx in xrange(args.tflen):
        tf_filename = os.path.join(args.tfdir, name_pat % ((args.tfstart + tf_indx)*obj_len, obj_len))
        print(tf_filename, os.path.isfile(tf_filename))

if __name__=='__main__':
    main()
