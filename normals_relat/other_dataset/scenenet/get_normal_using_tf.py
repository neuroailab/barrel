import tensorflow as tf
import argparse
import os
import numpy as np
import h5py

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
    parser.add_argument('--path', default = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/train/0/0/', type = str, action = 'store', help = 'Path to the directory hosting the depth and photo')
    parser.add_argument('--hdf5', default = '/home/chengxuz/barrel/barrel_github/dataset/scenenet/hdf5s/train_0.hdf5')

    args    = parser.parse_args()

    #filename_queue = tf.train.string_input_producer(
    #        tf.train.match_filenames_once(os.path.join(args.path, "*.jpg")))

    #indx_list = [0, 25, 50]
    batch_size = 50
    num_threads = 1
    batch_kwargs = {'batch_size': batch_size, 
            'num_threads': num_threads,
            'capacity': batch_size}

    indx_list = range(0, 7500, 25)
    depthname_list = [os.path.join(args.path, "depth",  "%i.png" % x) for x in indx_list]
    depths = get_batch_from_filename_list(depthname_list, decode_func = tf.image.decode_png, decode_func_kwargs = {'dtype': tf.uint16}, batch_kwargs = batch_kwargs)

    photoname_list = [os.path.join(args.path, "photo",  "%i.jpg" % x) for x in indx_list]
    photos = get_batch_from_filename_list(photoname_list, decode_func = tf.image.decode_jpeg, batch_kwargs = batch_kwargs, shape = (240, 320, 3))

    if args.gpu>-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    weight_np       = np.zeros([3, 3, 1, 3])
    weight_np[1, 0, 0, 0] = -0.5
    weight_np[1, 2, 0, 0] = 0.5
    weight_np[0, 1, 0, 1] = -0.5
    weight_np[2, 1, 0, 1] = 0.5
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

    # Start a new session to show example output.
    with tf.Session() as sess:
        # Required to get the filename matching to run.
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        #image_tensor = sess.run([image])
        image_tensor = sess.run([normals])
        #print(image_tensor)
        print(image_tensor[0].shape)
        print(type(image_tensor[0]))

        coord.request_stop()
        coord.join(threads)
