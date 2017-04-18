import tensorflow as tf
import argparse
import os
import numpy as np
import h5py
import time
import sys

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_data(normal_data, writer):
    for indx_now in xrange(normal_data.shape[0]):
        img = normal_data[indx_now]
        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())

def main():
    parser = argparse.ArgumentParser(description='The script to generate normal maps from depth images using tensorflow on GPU, for scenenet')
    parser.add_argument('--loaddir', default = '/mnt/fs1/Dataset/scenenet_combine/depth/', type = str, action = 'store', help = 'Directory of depth')
    parser.add_argument('--savedir', default = '/mnt/fs1/Dataset/scenenet_combine/normal_new/', type = str, action = 'store', help = 'Directory to store normals calculated')
    parser.add_argument('--indxsta', default = 0, type = int, action = 'store', help = 'Start index of file')
    parser.add_argument('--indxlen', default = 1, type = int, action = 'store', help = 'Number of files')

    args    = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(4)

    numb_perfile = 20

    depths = tf.placeholder(tf.float32, shape = (200, 240, 320, 1))
    
    weight_np       = np.zeros([3, 3, 1, 3])
    weight_np[1, 0, 0, 0] = 0.5
    weight_np[1, 2, 0, 0] = -0.5
    weight_np[0, 1, 0, 1] = 0.5
    weight_np[2, 1, 0, 1] = -0.5
    weight_conv2d = tf.constant(weight_np, dtype = tf.float32)

    bias_np         = np.zeros([3])
    bias_np[2]      = 1
    bias_tf         = tf.constant(bias_np, dtype = tf.float32)

    tmp_nor         = tf.nn.conv2d(depths, weight_conv2d,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    normals         = tf.nn.bias_add(tmp_nor, bias_tf)
    normals_n       = tf.nn.l2_normalize(normals, 3)
    normals_p       = tf.add(tf.multiply(normals_n, tf.constant(0.5)), tf.constant(0.5))

    with tf.Session() as sess:
        for file_indx in xrange(args.indxlen):
            curr_indx = file_indx + args.indxsta
            file_path = os.path.join(args.loaddir, 'data_%i_%i.tfrecords' % (curr_indx*20, numb_perfile))
            write_path = os.path.join(args.savedir, 'data_%i_%i.tfrecords' % (curr_indx*20, numb_perfile))

            record_iterator = tf.python_io.tf_record_iterator(path=file_path)
            writer = tf.python_io.TFRecordWriter(write_path)

            reconstructed_images = []
            batch_size = 200
            get_num = 0

            for string_record in record_iterator:
                
                example = tf.train.Example()
                example.ParseFromString(string_record)
                
                img_string = (example.features.feature['image_raw']
                                              .bytes_list
                                              .value[0])
                
                img_1d = np.fromstring(img_string, dtype=np.uint16)
                #img_1d = np.fromstring(img_string, dtype=np.uint8)
                reconstructed_img = img_1d.reshape((240, 320, -1))
                
                reconstructed_images.append(reconstructed_img)
                get_num = get_num + 1

                if get_num % batch_size==0:
                    batch_data = np.asarray(reconstructed_images)
                    batch_data = batch_data.astype(np.float)
                    #print batch_data.shape
                    normal_data = sess.run(normals_p, feed_dict={depths: batch_data})
                    normal_data = ( normal_data*255 ).astype(np.uint8)
                    print(normal_data.shape)

                    write_to_data(normal_data, writer)

                    reconstructed_images = []
            if len(reconstructed_images)>0:
                batch_data = np.asarray(reconstructed_images)
                batch_data = batch_data.astype(np.float)
                #print batch_data.shape
                normal_data = sess.run(normals_p, feed_dict={depths: batch_data})
                normal_data = ( normal_data*255 ).astype(np.uint8)
                print(normal_data.shape)

                write_to_data(normal_data, writer)

                reconstructed_images = []
            writer.close()

if __name__=="__main__":
    main()
