import re
import os
from scipy import misc
import numpy as np
from project_depth_map import *

def get_time_from_str(str_now):
    return float(str_now[2:].split('-')[0])

def get_synched_frames(sceneDir):

    ls_res = os.listdir(sceneDir)
    ls_res.sort()
    #print(len(ls_res))
    #print(ls_res[10])

    rgb_files   = []
    dep_files   = []
    for file_name in ls_res:
        if file_name.startswith('r-'):
            rgb_files.append(file_name)
        elif file_name.startswith('d-'):
            dep_files.append(file_name)

    #print(len(rgb_files))
    #print(len(dep_files))
    frame_files = []

    rgb_indx = 0
    for dep_file in dep_files:
        dep_time = get_time_from_str(dep_file)
        rgb_time = get_time_from_str(rgb_files[rgb_indx])

        t_diff = abs(rgb_time - dep_time)

        while rgb_indx < len(rgb_files)-1:
            rgb_time_tmp    = get_time_from_str(rgb_files[rgb_indx+1])
            t_diff_tmp      = abs(rgb_time_tmp - dep_time)

            if t_diff_tmp > t_diff:
                break

            t_diff          = t_diff_tmp
            rgb_indx        = rgb_indx + 1

        frame_files.append((os.path.join(sceneDir, dep_file), os.path.join(sceneDir, rgb_files[rgb_indx])))
    return frame_files

def main():
    sceneDir = '/Users/chengxuz/barrel/bullet/barrle_related_files/nyuv2/study_0005'

    host = os.uname()[1]
    if host =='kanefsky':
        sceneDir = '/home/chengxuz/barrel/barrel_github/dataset/nyuv2/nyuv2/study_0005'
    elif host =='freud':
        sceneDir = '/home/chengxuz/barrel/dataset/nyuv2/nyuv2/study_0005'
    elif host =='openmind7' or 'node' in host:
        sceneDir = '/om/user/chengxuz/Data/one_world_dataset/nyuv2/study_0005'

    frame_files = get_synched_frames(sceneDir)

    depth_test, rgb_test = frame_files[0]
    rgb_arr = misc.imread(rgb_test)

    dep_arr = misc.imread(depth_test)
    dep_arr = dep_arr.astype(np.uint16)
    dep_arr = dep_arr.byteswap()

    dep_mapped = project_depth_map(dep_arr, rgb_arr)

    pass

if __name__=='__main__':
    main()
