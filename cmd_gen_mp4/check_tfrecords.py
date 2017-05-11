import os
import numpy as np
import tensorflow as tf

'''
#savedir = '/om/user/chengxuz/Data/barrel_dataset/tfrecords'
savedir = '/scratch/users/chengxuz/barrel/barrel_relat_files/dataset/tfrecords'

key_list =[
    u'category',
    u'scale',
    u'Data_force',
    u'Data_normal',
    u'Data_torque',
    u'orn',
    u'position',
    u'speed'
    ]

objlen = 25
for key_now in key_list:
    checkdir = os.path.join(savedir, key_now)

    for objsta in xrange(0, 9981, objlen):
        path_now = os.path.join(checkdir, "Data%i_%i.tfrecords" % (objsta, objlen))

        if not os.path.isfile(path_now):
            print path_now
'''

#tfrecords_filename = '/media/data3/chengxuz/whisker/tfrecords/category/Data0_25.tfrecords'
tfrecords_filename = '/scratch/users/chengxuz/barrel/barrel_relat_files/dataset2/tfrecords/category/ctrain_100_0_21_strain.tfrecords'
reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
diff_name = []
diff_len = 144
#diff_len = 24
now_len = 0

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    img_string = (example.features.feature['category']
                                  .int64_list
                                  .value[0])
    
    #print(img_string)
    if now_len % diff_len==0:
        diff_name.append(img_string)
    else:
        if not diff_name[-1]==img_string:
            print('Error!')
    now_len = now_len + 1

print(diff_name)
print(now_len)
