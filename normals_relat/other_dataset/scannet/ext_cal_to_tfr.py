import tensorflow as tf
import argparse
import os
import numpy as np
import time
import sys
from scipy import misc
import cv2

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_tfrecs(arr_to_write, writer):
    for img_indx in xrange(arr_to_write.shape[0]):
        curr_img = arr_to_write[img_indx]

        img_raw = curr_img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())

def get_batch_by_imread(filename_list, indx_sta, indx_len, prep_func = None, prep_kwargs = {}, with_orig = False):
    indx_end = min(len(filename_list), indx_sta + indx_len)

    batch_imgs = None

    if with_orig:
        batch_orig = None

    for indx_img in xrange(indx_sta, indx_end):
        #print(indx_img)
        curr_img = misc.imread(filename_list[indx_img])

        if not prep_func is None:
            if with_orig:
                if batch_orig is None:
                    batch_orig = curr_img[np.newaxis, :]
                else:
                    batch_orig = np.concatenate([batch_orig, curr_img[np.newaxis, :]])
                
            curr_img = curr_img.astype(np.float32)
            curr_img = prep_func(curr_img, **prep_kwargs)

        if batch_imgs is None:
            batch_imgs = curr_img[np.newaxis, :]
        else:
            batch_imgs = np.concatenate([batch_imgs, curr_img[np.newaxis, :]])

    if len(batch_imgs.shape)==3:
        batch_imgs = batch_imgs[:, :, :, np.newaxis]
    if with_orig:
        return batch_orig, batch_imgs
    else:
        return batch_imgs

def main():
    parser = argparse.ArgumentParser(description='The script to generate normal maps from depth images using tensorflow on GPU or CPU, for scannet')
    parser.add_argument('--gpu', default = -1, type = int, action = 'store', help = 'Index of gpu to use, currently only one gpu is allowed')
    parser.add_argument('--path', default = '/om/user/chengxuz/Data/one_world_dataset/scannet/scannet/2016-08-06_02-59-01__C7BA9586-8237-4204-9116-02AE5804338A.sens', type = str, action = 'store', help = 'Path to the sens file including the photos and depths')
    parser.add_argument('--savedir', default = '/om/user/chengxuz/Data/one_world_dataset/scannet/tmpdir/d0', type = str, action = 'store', help = 'Path to the temporary directory hosting the depth and photo')
    parser.add_argument('--tfrdir', default = '/om/user/chengxuz/Data/one_world_dataset/scannet/tfrecs/', type = str, action = 'store', help = 'Path to save the tfrecs, subfolders will be created')
    parser.add_argument('--tfrname', default = 'scannet0.tfrecord', type = str, action = 'store', help = 'Name of the tfrec file')
    parser.add_argument('--exepath', default = '/om/user/chengxuz/barrel/ScanNet/SensReader/sens', type = str, action = 'store', help = 'Path for the SenseReader')

    args    = parser.parse_args()

    # Do the extraction

    cmd_ext = '%s %s %s'
    os.system(cmd_ext % (args.exepath, args.path, args.savedir))

    # Calculate the normals

    batch_size = 20

    file_list = os.listdir(args.savedir)
    frame_num = (len(file_list) - 1)/3
    depthname_list = [os.path.join(args.savedir, 'frame-%06i.depth.pgm' % x) for x in xrange(frame_num)]
    #depthname_list = [os.path.join('/om/user/chengxuz/Data/one_world_dataset/split_scenenet/train/0/0/depth', x) for x in os.listdir('/om/user/chengxuz/Data/one_world_dataset/split_scenenet/train/0/0/depth')]
    #photoname_list = [os.path.join(args.savedir, 'frame-%06i.color.jpg' % x) for x in xrange(frame_num)]

    bilat_kwargs = {'d': -1, 'sigmaColor': 100, 'sigmaSpace': 2}

    #depth_tfrec_name = os.path.join(args.tfrdir, 'depths', args.tfrname)
    #depth_writer = tf.python_io.TFRecordWriter(depth_tfrec_name)

    normal_tfrec_name = os.path.join(args.tfrdir, 'normals', args.tfrname)
    normal_writer = tf.python_io.TFRecordWriter(normal_tfrec_name)

    #photo_tfrec_name = os.path.join(args.tfrdir, 'images', args.tfrname)
    #photo_writer = tf.python_io.TFRecordWriter(photo_tfrec_name)

    #depthname_list = [os.path.join(args.savedir, 'frame-%06i.png' % x) for x in xrange(frame_num)]
    #photoname_list = [os.path.join(args.savedir, 'frame-%06i.jpg' % x) for x in xrange(frame_num)]

    #depth_shape = (480, 640)
    #photo_shape = (968, 1296, 3)

    depths = tf.placeholder(tf.float32)

    if args.gpu>-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

    # Save to tfrecs

    with tf.Session() as sess:

        for indx_now in xrange(0, len(depthname_list), batch_size):
            start_time = time.time()
            #depths_org, depths_arr = get_batch_by_imread(depthname_list, indx_now, batch_size, cv2.bilateralFilter, bilat_kwargs, True)
            depths_arr = get_batch_by_imread(depthname_list, indx_now, batch_size, cv2.bilateralFilter, bilat_kwargs, False)
            load_time = time.time()
            #photos_arr = get_batch_by_imread(photoname_list, indx_now, batch_size)

            #depths_org = depths_org.astype(np.uint16)
            #photos_arr = photos_arr.astype(np.uint8)

            normal_tensor = sess.run([normals_p], feed_dict = {depths: depths_arr})

            normal_arr = (255*normal_tensor[0]).astype(np.uint8)

            #write_to_tfrecs(depths_org, depth_writer)
            #write_to_tfrecs(photos_arr, photo_writer)
            write_to_tfrecs(normal_arr, normal_writer)
            #print(normal_arr.shape)
            #print(normal_arr[0,0,0])
            #print(photos_arr.shape)

            #print(len(normal_tensor))
            #print(normal_tensor[0].shape)
            #print(normal_tensor[0][0,0,0])
            #break
            end_time = time.time()
            #end_time = time.time()

            print("%i Batch time: %f, load time: %f" % (indx_now//batch_size, end_time - start_time, load_time - start_time))
            sys.stdout.flush()

    #depth_writer.close()
    normal_writer.close()
    #photo_writer.close()

    # Clear the generated images
    cmd_rm = 'rm -r %s'
    os.system(cmd_rm % args.savedir)

if __name__=="__main__":
    main()
