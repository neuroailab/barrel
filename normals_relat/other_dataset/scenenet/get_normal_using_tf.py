import tensorflow as tf
import argparse
import os
import numpy as np
import h5py
import time
import sys

def get_batch_from_filename_list(filename_list, decode_func, batch_kwargs, decode_func_kwargs = {}, shuffle = False, shape = (240, 320, 1)):
    filename_queue = tf.train.string_input_producer(
            filename_list, shuffle = shuffle)

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = decode_func(image_file, **decode_func_kwargs)

    image.set_shape(shape)

    image_float     = tf.cast(image, tf.float32)

    images = tf.train.batch(
        [image_float],
        **batch_kwargs)

    return images

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to generate normal maps from depth images using tensorflow on GPU, for scenenet')
    parser.add_argument('--gpu', default = -1, type = int, action = 'store', help = 'Index of gpu to use, currently only one gpu is allowed')
    parser.add_argument('--path', default = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/train/0/', type = str, action = 'store', help = 'Path to the directory hosting the depth and photo')
    parser.add_argument('--hdf5', default = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/hdf5s/train_0.hdf5')
    parser.add_argument('--seed', default = 0, type = int, action = 'store', help = 'Random seed used to generate random permutations')

    args    = parser.parse_args()

    #filename_queue = tf.train.string_input_producer(
    #        tf.train.match_filenames_once(os.path.join(args.path, "*.jpg")))

    #indx_list = [0, 25, 50]
    #batch_size = 10
    #batch_size = 70
    batch_size = 256
    num_threads = 5
    batch_kwargs = {'batch_size': batch_size, 
            'num_threads': num_threads,
            'capacity': batch_size}

    #file_list = range(10)
    #file_list = range(1000)
    file_list = os.listdir(args.path)
    indx_list = range(0, 7500, 25)
    findx_list = [(int(x), y) for x in file_list for y in indx_list]

    np.random.seed(args.seed)
    findx_list = np.random.permutation(findx_list)

    depthname_list = [os.path.join(args.path, str(x), "depth",  "%i.png" % y) for x,y in findx_list]
    photoname_list = [os.path.join(args.path, str(x), "photo",  "%i.jpg" % y) for x,y in findx_list]

    depths = get_batch_from_filename_list(depthname_list, decode_func = tf.image.decode_png, decode_func_kwargs = {'dtype': tf.uint16}, batch_kwargs = batch_kwargs)
    photos = get_batch_from_filename_list(photoname_list, decode_func = tf.image.decode_jpeg, batch_kwargs = batch_kwargs, shape = (240, 320, 3))

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

    # Set hdf5 related files
    with h5py.File(args.hdf5, 'a') as h5_file:

        if 'images' in h5_file:
            if not h5_file['images'].shape[0]==len(findx_list):
                del h5_file['images']
        if 'normals' in h5_file:
            if not h5_file['normals'].shape[0]==len(findx_list):
                del h5_file['normals']

        if 'images' not in h5_file:
            #h5_images = h5_file.create_dataset('images', (len(findx_list), 240, 320, 3), dtype='uint8', compression="gzip")
            h5_images = h5_file.create_dataset('images', (len(findx_list), 240, 320, 3), dtype='uint8')
        else:
            h5_images = h5_file['images']
        if 'normals' not in h5_file:
            #h5_normals = h5_file.create_dataset('normals', (len(findx_list), 240, 320, 3), dtype='uint8', compression="gzip")
            h5_normals = h5_file.create_dataset('normals', (len(findx_list), 240, 320, 3), dtype='uint8')
        else:
            h5_normals = h5_file['normals']

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for indx_now in xrange(0, len(findx_list), batch_size):
                start_time = time.time()
                image_tensor, normal_tensor = sess.run([photos, normals_p])
                if (indx_now+batch_size <= len(findx_list)):
                    curr_add_num = batch_size
                else:
                    curr_add_num = len(findx_list) - indx_now

                gpu_time = time.time()
                h5_images[indx_now:indx_now+curr_add_num] = image_tensor[:curr_add_num].astype(np.uint8)
                h5_normals[indx_now:indx_now+curr_add_num] = (255*normal_tensor[:curr_add_num]).astype(np.uint8)
                end_time = time.time()

                print("%i Batch time: %f, gpu time: %f, save time: %f" % (indx_now//batch_size, end_time - start_time, gpu_time - start_time, end_time - gpu_time))
                sys.stdout.flush()

            coord.request_stop()
            coord.join(threads)
