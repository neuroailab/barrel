import h5py
import numpy as np
from PIL import Image
import os

#file_name = os.path.join('/Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s', 'test_7579466178976215500_1_0_0_2.hdf5')
#file_name = os.path.join('/Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s', 'aftervhacd_-3199863574928696085_1_0_0_2.hdf5')
#file_name = os.path.join('/Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s', 'teddy_-7904248716561876032_1_0_0_2.hdf5')
#file_name = os.path.join('/Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s', 'teddy_-7234480266557311556_1_3_0_2.hdf5')
#file_name = os.path.join('/Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s', 'teddy_-7904248716561876032_1_0_0_2.hdf5')
#file_name = os.path.join('/Users/chengxuz/barrel/bullet/barrle_related_files/hdf5s', 'aftervhacd_4793304094075214567_1_3_0_2.hdf5')
#file_name = os.path.join('/home/chengxuz/barrel/related_files/hdf5_files', 'aftervhacd_5176502897631907425_0_0_0_0.hdf5')
#file_name = os.path.join('/home/chengxuz/barrel/related_files/hdf5_files', 'duck_1202383454118148709_0_0_0_0.hdf5')
#file_name = os.path.join('/home/chengxuz/barrel/related_files/hdf5_files', 'teddy_-7879274547486381024_0_0_0_0.hdf5')
file_name = os.path.join('/home/chengxuz/barrel/related_files/hdf5_files', 'duck_-1083721545439873894_0_0_0_0.hdf5')

now_im_indx = 0

fin = h5py.File(file_name, 'r')

for now_im_indx in xrange(5):
    #save_im_name = 'afterhat_%s.png' % now_im_indx
    save_im_name = 'duck_%s.png' % now_im_indx
    #save_im_name = 'teddy_%s.png' % now_im_indx
    normal = np.asarray(fin['Data_normal'][now_im_indx])

    result = Image.fromarray((normal * 255).astype(np.uint8))

    result.save(save_im_name)
