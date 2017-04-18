import h5py
import numpy as np
import cPickle

path_in = '/data/chengxuz/barrel_response/response.hdf5'
#path_cpickle = '/data/chengxuz/barrel_response/response_fcadd.pkl'
path_cpickle = '/data/chengxuz/barrel_response/response_label.pkl'

fin = h5py.File(path_in)

all_label = fin['label'][...]
all_fcadd = fin['fc_add_0'][...]
#cPickle.dump(all_fcadd, open(path_cpickle, 'w'))
cPickle.dump(all_label, open(path_cpickle, 'w'))

argmax_fcadd = np.argmax(all_fcadd, 1)

correct_flag = all_label==argmax_fcadd

num_cate = 117
cate_result = []

for indx_cate in xrange(num_cate):
    num_corr_now = np.sum(all_label[correct_flag]==indx_cate)
    num_all_now = np.sum(all_label==indx_cate)
    corr_rate_now = num_corr_now*1.0/num_all_now
    cate_result.append((corr_rate_now, indx_cate))


cate_result.sort()
print(cate_result)
