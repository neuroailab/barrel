import tensorflow as tf
import numpy as np
import time

#tfrecords_filename = '/data/chengxuz/whisker/tfrecs_all/tfrecords/Data_force/Data%i_25.tfrecords'
tfrecords_filename = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords/Data_force/Data%i_25.tfrecords'
reconstructed_images = []

file_offset = 30 
indx_list = np.random.permutation(range(30))

for file_indx in indx_list:
    tf_path = tfrecords_filename % (file_indx*25 + file_offset*25)
    record_iterator = tf.python_io.tf_record_iterator(path=tf_path)

    time_now = 0

    pre_time = time.time()

    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        '''
        
        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])
        
        #img_1d = np.fromstring(img_string, dtype=np.uint16)
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((240, 320, -1))
        
        reconstructed_images.append(reconstructed_img)
        '''
        time_now = time.time() - pre_time + time_now
        pre_time = time.time()

    print(time_now)

