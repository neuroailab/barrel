import os
import numpy as np
import argparse
import cPickle

def main():
    parser = argparse.ArgumentParser(description='Combine the statistics')
    #parser.add_argument('--saveprefix', default = '/om/user/chengxuz/Data/barrel_dataset/statistics/Data_force_', type = str, action = 'store', help = 'Name prefix for the saved pkl')
    #parser.add_argument('--saveprefix', default = '/om/user/chengxuz/Data/barrel_dataset/statistics/Data_torque_', type = str, action = 'store', help = 'Name prefix for the saved pkl')
    parser.add_argument('--saveprefix', default = '/scratch/users/chengxuz/barrel/barrel_relat_files/dataset/statistic/Data_force_', type = str, action = 'store', help = 'Name prefix for the saved pkl')
    #parser.add_argument('--savepath', default = '/om/user/chengxuz/Data/barrel_dataset/statistics/Data_force_combined.pkl', type = str, action = 'store', help = 'Path for saving the computed statistics')
    #parser.add_argument('--savepath', default = '/om/user/chengxuz/Data/barrel_dataset/statistics/Data_torque_combined.pkl', type = str, action = 'store', help = 'Path for saving the computed statistics')
    parser.add_argument('--savepath', default = '/scratch/users/chengxuz/barrel/barrel_relat_files/dataset/statistic/Data_torque_combined.pkl', type = str, action = 'store', help = 'Path for saving the computed statistics')

    args    = parser.parse_args()

    shape_tfr = [110, 31, 3, 3]
    file_len = 20

    sum_array = np.zeros(shape_tfr)
    sum_sq_array = np.zeros(shape_tfr)
    max_array = np.zeros(shape_tfr)
    min_array = np.zeros(shape_tfr)
    num_add = 0

    for file_indx in xrange(0, 399, file_len):
        name_tmp = args.saveprefix + 'sta%i_len%i.pkl' % (file_indx, file_len)
        data_tmp = cPickle.load(open(name_tmp, 'r'))

        sum_array = data_tmp['sum_array'] + sum_array
        sum_sq_array = data_tmp['sum_sq_array'] + sum_sq_array
        num_add = data_tmp['num_add'] + num_add

        max_array = np.maximum(data_tmp['max_array'], max_array)
        min_array = np.minimum(data_tmp['min_array'], min_array)

        '''
        tmp_array = sum_array.reshape([110*31, 3, 3])
        mean_now = np.mean(tmp_array, 0)/num_add
        print(mean_now)
        tmp_sq_array = sum_sq_array.reshape([110*31, 3, 3])
        std_now = np.sqrt(np.mean(tmp_sq_array, 0)/num_add - mean_now**2)
        print(std_now)
        '''
        tmp_array = data_tmp['sum_array'].reshape([110*31, 3, 3])
        mean_now = np.mean(tmp_array, 0)/num_add
        print(mean_now)
        tmp_sq_array = data_tmp['sum_sq_array'].reshape([110*31, 3, 3])
        std_now = np.sqrt(np.mean(tmp_sq_array, 0)/num_add - mean_now**2)
        print(std_now)
        print(name_tmp)

    tmp_max = max_array.reshape([110*31, 3, 3])
    print(np.max(tmp_max, 0))
    tmp_min = min_array.reshape([110*31, 3, 3])
    print(np.min(tmp_min, 0))

    save_dict = {}
    save_dict['mean'] = mean_now
    save_dict['std'] = std_now
    cPickle.dump(save_dict, open(args.savepath, 'w'))

if __name__=='__main__':
    main()
