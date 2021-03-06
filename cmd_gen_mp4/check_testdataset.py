import os
import h5py
import numpy as np
import sys

'''
hdf_dir = "/scratch/users/chengxuz/barrel/barrel_relat_files/testdataset_diver"
bigsamnum = 48
seed_len = 251

for seed_now in xrange(seed_len):
    for now_samnum in xrange(bigsamnum):
        for name in ['teddy', 'duck']:
            now_path = os.path.join(hdf_dir, "Data%i_%i_%s_%i.hdf5" % (seed_now, now_samnum, name, seed_now))
            if not os.path.exists(now_path):
                print(now_path)

'''
if False:
    hdf_dir = "/om/user/chengxuz/barrel/testdataset_dis"

    now_num = 0

    obj_name_list = ['duck', 'teddy']

    for obj_now in obj_name_list:
        for pos_now in xrange(36):
            for scale_now in xrange(6):
                for speed_now in xrange(14):
                    for orn_now in xrange(15):
                        now_path = os.path.join(hdf_dir, "%s_%i_%i_%i_%i.hdf5" % (obj_now, pos_now, scale_now, speed_now, orn_now))
                        run_flag = False
                        if not os.path.exists(now_path):
                            print(now_path, 'Not existing!')
                            run_flag = True
                            #continue
                        '''
                        else:
                            fin = h5py.File(now_path, 'r')
                            scale = fin['Data']['scale'][...]
                            if obj_now=='teddy' and scale[0]/(scale[3]/30)>1:
                                print(now_path, 'Wrong data!')
                                run_flag = True
                            if obj_now=='duck' and scale[0]/(scale[3]/30)<1:
                                print(now_path, 'Wrong data!')
                                run_flag = True
                        '''
                            
                        if run_flag:
                            os.system('sbatch script_sbatch_hdf5_om.sh %i %i %i %i %i' % (obj_name_list.index(obj_now), pos_now, scale_now, speed_now, orn_now))
                        now_num = now_num + 1
                        if now_num%1000==0:
                            print('Now file %i!' % now_num)
                        sys.stdout.flush()

#curr_control = 1
#curr_control = 2
#curr_control = 3
curr_control = 4
hdf_dir = "/om/user/chengxuz/barrel/testdataset_control%i" % curr_control
bigsamnum = 96
seed_len = 125

non_num = 0
cmd1_str = 'sbatch script_sbatch_hdf5_om.sh /om/user/chengxuz/barrel/bullet3/data/teddy2_VHACD_CHs.obj %i teddy_%i %i %i %i'
#        sbatch script_sbatch_hdf5_om.sh ${data_path2} ${bigsamnum} duck_${seed} ${seed} ${control}
cmd2_str = 'sbatch script_sbatch_hdf5_om.sh /om/user/chengxuz/barrel/bullet3/data/duck_vhacd.obj %i duck_%i %i %i %i'

for seed_now in xrange(seed_len):
    for now_samnum in xrange(bigsamnum):
        for name,cmd_str in [('teddy', cmd1_str), ('duck', cmd2_str)]:
            now_path = os.path.join(hdf_dir, "Data%i_%i_%s_%i.hdf5" % (seed_now, now_samnum, name, seed_now))
            if not os.path.exists(now_path):
                print(now_path)
                act_cmd_str = cmd_str % (bigsamnum, seed_now, seed_now, curr_control, now_samnum)
                print(act_cmd_str)
                os.system(act_cmd_str)
                #exit()
                non_num = non_num + 1

print(non_num)
