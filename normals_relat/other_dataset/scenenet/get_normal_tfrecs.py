import tensorflow as tf
import argparse
import os
import numpy as np
import h5py
import time
import sys
import multiprocessing

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

findx_list = []
args = None
fin = None

def write_it(ind):
    global args
    global findx_list
    global fin

    indx_len = 300
    indx_list = [findx_list.index(ind*indx_len + indx) for indx in xrange(indx_len)]

    tfrecords_filename = os.path.join(args.path, 'data_%i.tfrecords' % (ind + args.seed*1000))

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for indx_now in indx_list:
        img = np.array(fin['normals'][indx_now])
            
        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())

    writer.close()

def main():

    global findx_list
    global args
    global fin
    
    parser = argparse.ArgumentParser(description='The script to change normal maps from hdf5s to tfrecords, for scenenet')
    parser.add_argument('--path', default = '/om/user/chengxuz/Data/one_world_dataset/split_scenenet/train_tfrecords/0/normal', type = str, action = 'store', help = 'Path to the directory hosting the depth and photo')
    parser.add_argument('--hdf5', default = '/om/user/chengxuz/Data/one_world_dataset/split_scenenet/hdf5s/train_0.hdf5')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed used to generate random permutations')

    args    = parser.parse_args()

    os.system('mkdir -p %s' % args.path)

    fin = h5py.File(args.hdf5, 'r')
    len_file = fin['images'].shape[0]

    indx_list = range(0, 7500, 25)
    file_list = range(args.seed*1000, args.seed*1000 + len_file/len(indx_list))
    findx_list = [(x, y) for x in file_list for y in indx_list]

    np.random.seed(args.seed)
    findx_list = list(np.random.permutation(len(findx_list)))

    nproc = 5
    pool = multiprocessing.Pool(processes=nproc)
    r = pool.map_async(write_it, range(len_file/len(indx_list)))
    #r = pool.map_async(write_it, range(5))
    #r = pool.map_async(write_it, range(min(file_list), min(file_list) + 5))
    r.get()
    print('Done!')

    fin.close()

if __name__=="__main__":
    main()
