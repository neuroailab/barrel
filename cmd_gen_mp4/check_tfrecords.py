import os
import numpy as np

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
