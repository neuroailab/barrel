import tensorflow as tf
import argparse
import os
import numpy as np
import h5py
import time
import sys
import multiprocessing

args = None

def write_it(ind):
    global args

    tfrecords_filename = os.path.join(args.savedir, 'data_%i_%i.tfrecords' % (ind, args.combinelen))

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for indx_now in xrange(ind, ind + args.combinelen):
        curr_path = os.path.join(args.loaddir, 'data_%i.tfrecords' % indx_now)
        if not os.path.isfile(curr_path):
            break

        record_iterator = tf.python_io.tf_record_iterator(path=curr_path)
        for string_record in record_iterator:
            writer.write(string_record)

    writer.close()

def main():
    global args

    parser = argparse.ArgumentParser(description='The script to combine tfrecords files for scenenet')
    parser.add_argument('--loaddir', default = '/mnt/fs1/Dataset/scenenet/photo', type = str, action = 'store', help = 'Path to the directory hosting the original tfrecords')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/scenenet_combine/photo')
    parser.add_argument('--indxsta', default = 0, type = int, action = 'store', help = 'Start index of combine file')
    parser.add_argument('--indxlen', default = 1, type = int, action = 'store', help = 'Number of big combining to do')
    parser.add_argument('--combinelen', default = 20, type = int, action = 'store', help = 'Length of tfrecords files to be included in one file')

    args    = parser.parse_args()

    os.system('mkdir -p %s' % args.savedir)

    file_list = os.listdir(args.loaddir)
    file_len = len(file_list) - 1 # remove meta.pkl file

    start_indx = args.indxsta * args.combinelen
    end_indx = min(start_indx + args.indxlen * args.combinelen, file_len)

    nproc = 5
    pool = multiprocessing.Pool(processes=nproc)
    r = pool.map_async(write_it, xrange(start_indx, end_indx, args.combinelen))
    r.get()
    print('Done!')

if __name__=="__main__":
    main()
