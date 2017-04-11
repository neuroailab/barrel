import h5py
import numpy as np

hdf5_name = '/om/user/chengxuz/Data/barrel_dataset/raw_hdf5/Data4039_9_6e5bf008a9259e95fa80fb391ee7ccee.hdf5'

fin = h5py.File(hdf5_name, 'r')
key_list = ['Data_force', 'Data_torque']

for which_data in xrange(12):
    which_entry = fin['Data%i' % which_data]
    for key_now in key_list:
        arr_now = which_entry[key_now][...]
        print(np.std(arr_now))
