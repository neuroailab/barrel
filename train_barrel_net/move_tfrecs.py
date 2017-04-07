import os

from_dir = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_sher/tfrecords'
to_dir = '/mnt/fs0/chengxuz/Data/whisker/tfrecs_all/tfrecords'

#print(os.listdir(from_dir))

for sub_dir in os.listdir(from_dir):
    tmp_from_dir = os.path.join(from_dir, sub_dir)
    tmp_to_dir = os.path.join(to_dir, sub_dir)

    mv_list = os.listdir(tmp_from_dir)
    for file_name in mv_list:
        if file_name=='meta.pkl':
            continue
        new_name = 'sher_' + file_name
        new_path = os.path.join(tmp_to_dir, new_name)
        old_path = os.path.join(tmp_from_dir, file_name)
        cmd_str = 'mv %s %s' % (old_path, new_path)
        print(cmd_str)
        os.system(cmd_str)
    #break
