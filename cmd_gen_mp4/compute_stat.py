import os
import numpy as np
import tensorflow as tf
import argparse
import cPickle

def main():
    parser = argparse.ArgumentParser(description='Compute the statistics of the input in order to normalize')
    parser.add_argument('--tflen', default = 1, type = int, action = 'store', help = 'Number of tfrecs for computing')
    parser.add_argument('--tfstart', default = 0, type = int, action = 'store', help = 'Start index of tfrecs for computing')
    parser.add_argument('--tfdir', default = '/om/user/chengxuz/Data/barrel_dataset/tfrecords/', type = str, action = 'store', help = 'Directory holding the tfrec files')
    parser.add_argument('--tfkey', default = 'Data_force', type = str, action = 'store', help = 'Key for the tfrecords')
    parser.add_argument('--saveprefix', default = '/om/user/chengxuz/Data/barrel_dataset/statistics/Data_force_', type = str, action = 'store', help = 'Name prefix for the saved pkl')

    args    = parser.parse_args()

    shape_tfr = [110, 31, 3, 3]
    obj_len = 25 # Number of objs in one tfrecord file
    #name_pat = 'Data%i_%i.tfrecords'
    name_pat = 'sher_Data%i_%i.tfrecords'

    sum_array = np.zeros(shape_tfr)
    sum_sq_array = np.zeros(shape_tfr)
    max_array = np.zeros(shape_tfr)
    min_array = np.zeros(shape_tfr)
    num_add = 0

    for tf_indx in xrange(args.tflen):
        tf_filename = os.path.join(args.tfdir, args.tfkey, name_pat % ((args.tfstart + tf_indx)*obj_len, obj_len))
        print(tf_filename)

        #if 'Data4025_25.tfrecords' in tf_filename:
        #    continue

        record_iterator = tf.python_io.tf_record_iterator(path=tf_filename)

        #print(tf_filename, os.path.isfile(tf_filename))
        #reconstructed_images = []
    
        for string_record in record_iterator:
            
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            img_string = (example.features.feature[args.tfkey]
                                          .bytes_list
                                          .value[0])
            
            #img_1d = np.fromstring(img_string, dtype=np.uint16)
            img_1d = np.fromstring(img_string, dtype=np.float32)
            reconstructed_img = img_1d.reshape(shape_tfr)
            
            #reconstructed_images.append(reconstructed_img)
            num_add = num_add + 1
            sum_array = sum_array + reconstructed_img
            sum_sq_array = sum_sq_array + reconstructed_img**2

            max_array = np.maximum(max_array, reconstructed_img)
            min_array = np.minimum(min_array, reconstructed_img)

            tmp_mean = np.mean(reconstructed_img)
            tmp_std = np.std(reconstructed_img)
            if tmp_std>10000000:
                print('Error data here %i with %f!' % (num_add, tmp_std))
        #print(num_add)
        #print(sum_array.shape)
        #break
        tmp_array = sum_array.reshape([110*31, 3, 3])
        mean_now = np.mean(tmp_array, 0)/num_add
        #print(mean_now)
        tmp_sq_array = sum_sq_array.reshape([110*31, 3, 3])
        std_now = np.sqrt(np.mean(tmp_sq_array, 0)/num_add - mean_now**2)
        #print(std_now)

        #tmp_max = max_array.reshape([110*31, 3, 3])
        #print(np.max(tmp_max, 0))
        #tmp_min = min_array.reshape([110*31, 3, 3])
        #print(np.min(tmp_min, 0))

    save_dict = {}
    save_dict['num_add'] = num_add
    save_dict['sum_array'] = sum_array
    save_dict['sum_sq_array'] = sum_sq_array
    save_dict['max_array'] = max_array
    save_dict['min_array'] = min_array

    #save_pathname = args.saveprefix + 'sta%i_len%i.pkl' % (args.tfstart, args.tflen)
    #cPickle.dump(save_dict, open(save_pathname, 'w'))

if __name__=='__main__':
    main()
