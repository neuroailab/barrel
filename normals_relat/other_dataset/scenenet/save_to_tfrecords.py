import tensorflow as tf
import argparse
import os
import numpy as np
import sys
from PIL import Image
import multiprocessing

args    = None


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_it(ind):
    global args

    indx_list = range(0, 7500, 25)

    if args.mode==0:
        photoname_list = [os.path.join(args.path, str(ind), "photo",  "%i.jpg" % y) for y in indx_list]
    elif args.mode==1:
        photoname_list = [os.path.join(args.path, str(ind), "depth",  "%i.png" % y) for y in indx_list]
    elif args.mode==2:
        photoname_list = [os.path.join(args.path, str(ind), "instance",  "%i.png" % y) for y in indx_list]

    photoname_list = filter(lambda x: os.path.isfile(x), photoname_list)

    tfrecords_filename = os.path.join(args.savedir, 'data_%i.tfrecords' % ind)

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_path in photoname_list:

        img = np.array(Image.open(img_path))

        if args.mode>0:
            img = img.astype(np.uint16)
            
        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())

    writer.close()

def main():
    global args

    parser = argparse.ArgumentParser(description='The script to write to tfrecords from original images using tensorflow, for scenenet')
    parser.add_argument('--path', default = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/train/0/', type = str, action = 'store', help = 'Path to the directory hosting the depth and photo')
    #parser.add_argument('--savedir', default = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords_depth')
    parser.add_argument('--savedir', default = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/tfrecords_ins')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed used to generate random permutations')
    parser.add_argument('--mode', default = 0, type = int, action = 'store', help = '0 for photo, 1 for depth, and 2 for instance')

    args    = parser.parse_args()

    file_list = os.listdir(args.path)

    # tmp code
    #file_list = file_list[:1]
    file_list = filter(lambda x: x.isdigit(), file_list)
    file_list = [int(x) for x in file_list]
    print(len(file_list))

    os.system('mkdir -p %s' % args.savedir)

    #np.random.seed(args.seed)
    #findx_list = np.random.permutation(findx_list)

    nproc = 5
    pool = multiprocessing.Pool(processes=nproc)
    r = pool.map_async(write_it, range(min(file_list), max(file_list)))
    #r = pool.map_async(write_it, range(min(file_list), min(file_list) + 5))
    r.get()
    print('Done!')

if __name__=="__main__":
    main()
