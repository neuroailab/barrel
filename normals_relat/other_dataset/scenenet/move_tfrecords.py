# script for gathering tfrecords
import os

#from_dir = '/mnt/fs1/chengxuz/normal_data/train_tfrecords'
from_dir = '/mnt/fs1/chengxuz/normal_data/scenenet/normals/train_tfrecords'
to_dir = '/mnt/fs1/Dataset/scenenet'

deal_dirs = os.listdir(from_dir)
print(len(deal_dirs))
#attr_move = ['depth', 'instance', 'photo']
attr_move = ['normal']
for attr_now in attr_move:
    to_attr_dir = os.path.join(to_dir, attr_now)
    os.system('mkdir -p %s' % to_attr_dir)
    for deal_dir in deal_dirs:
        from_attr_dir = os.path.join(from_dir, deal_dir, attr_now)
        cmd_str = 'mv %s/* %s' % (from_attr_dir, to_attr_dir)
        print(cmd_str)
        os.system(cmd_str)
        #break
    #break
