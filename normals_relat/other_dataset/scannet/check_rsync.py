import os
import time

check_dir = '/mnt/data2/chengxuz/scannet'

rsync_cmd = 'rsync -avzh %s chengxuz@openmind7.mit.edu:/om/user/chengxuz/Data/one_world_dataset/scannet/scannet/'

PAUSE_TIME = 3

finish_dict = []
wait_list = {}

end_count = 0

while True:
    file_list = os.listdir(check_dir)

    for file_now in file_list:
        if file_now in finish_dict:
            continue
        size_now = os.path.getsize(os.path.join(check_dir, file_now))
        if file_now in wait_list:
            if size_now==wait_list[file_now]['size']:
                wait_list[file_now]['equal'] = wait_list[file_now]['equal'] + 1
            else:
                wait_list[file_now]['size'] = size_now
                wait_list[file_now]['equal'] = 0
            continue
        wait_list[file_now] = {}
        wait_list[file_now]['size'] = size_now
        wait_list[file_now]['equal'] = 0

    if len(wait_list)==0:
        end_count = end_count + 1
    else:
        end_count = 0

    if end_count==3:
        break

    pop_list = []
    for file_now in wait_list:
        if wait_list[file_now]['equal']==3:
            os.system(rsync_cmd % os.path.join(check_dir, file_now))

            pop_list.append(file_now)
            finish_dict.append(file_now)

    for file_now in pop_list:
        wait_list.pop(file_now)

    time.sleep(PAUSE_TIME)
