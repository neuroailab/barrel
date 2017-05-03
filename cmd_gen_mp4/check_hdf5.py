import h5py
import numpy as np
import os

key_list = ['Data_force', 'Data_torque']
'''
hdf5_name = '/om/user/chengxuz/Data/barrel_dataset/raw_hdf5/Data4039_9_6e5bf008a9259e95fa80fb391ee7ccee.hdf5'

fin = h5py.File(hdf5_name, 'r')

for which_data in xrange(12):
    which_entry = fin['Data%i' % which_data]
    for key_now in key_list:
        arr_now = which_entry[key_now][...]
        print(np.std(arr_now))
'''

#hdf5_folder = '/om/user/chengxuz/Data/barrel_dataset2/raw_hdf5/'
hdf5_folder = '/om/user/chengxuz/Data/barrel_dataset/raw_hdf5/'
file_list = os.listdir(hdf5_folder)

file_num = 0

for file_name in file_list:
    #file_path = os.path.join(hdf5_folder, file_name)
    file_path = '/om/user/chengxuz/Data/barrel_dataset/raw_hdf5/Data4030_0_5776239d3e133b68a02fa8dacc37a297.hdf5'

    fin = h5py.File(file_path, 'r')

    for which_data in fin:
        which_entry = fin[which_data]
        for key_now in key_list:
            arr_now = which_entry[key_now][...]
            print(np.std(arr_now))
            max_tmp = np.max(arr_now)
            if max_tmp < 100000:
                print(max_tmp)
            else:
                print(max_tmp, file_path, which_data, key_now)
            

    file_num = file_num + 1
    #if file_num>100:
    if file_num>0:
        break

