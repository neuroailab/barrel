import argparse
import os
from itertools import *
import copy
from cmd_gen import *
import numpy as np

def get_speed_list(mode=0):
    #return [[0,-12,0], [0, -10, 0], [0, -8, 0]]
    if mode==0:
        return [[0,-12,0], [0, -10, 0], [0, -8, 0]]
    elif mode==1:
        return [[0,-12.5,0], [0, -10.5, 0], [0, -8.5, 0]]
    elif mode==2:
        start_speed = -14
        end_speed = -7

        ret_list = []

        for now_speed in np.arange(start_speed, end_speed):
            now_speed_list = [0, now_speed, 0]
            ret_list.append(now_speed_list)

        start_speed = -14.5
        end_speed = -7.5

        for now_speed in np.arange(start_speed, end_speed):
            now_speed_list = [0, now_speed, 0]
            ret_list.append(now_speed_list)

        return ret_list
    else:
        return [[0,-12,0], [0, -10, 0], [0, -8, 0]]

def qua_from_euler(euler):
    cos_x = np.cos(euler[0]/2)
    sin_x = np.sin(euler[0]/2)
    cos_y = np.cos(euler[1]/2)
    sin_y = np.sin(euler[1]/2)
    cos_z = np.cos(euler[2]/2)
    sin_z = np.sin(euler[2]/2)

    qua_0 = cos_x*cos_y*cos_z + sin_x*sin_y*sin_z
    qua_1 = sin_x*cos_y*cos_z - cos_x*sin_y*sin_z
    qua_2 = cos_x*sin_y*cos_z + sin_x*cos_y*sin_z
    qua_3 = cos_x*cos_y*sin_z - sin_x*sin_y*cos_z

    return [qua_0, qua_1, qua_2, qua_3]

def get_orn_list(mode=0):
    if mode==0:
        return [[0,0,0,1], [1,0,0,0], [0,1,1,1], [1,1,1,1]]
    elif mode==1:
        return [[0,0,0,1], [1,0,0,0], [0,1,1,1], [1,1,1,1]]
    elif mode==2:
        ret_val = []
        offset_unit = (5.0/180.0)*np.pi

        for offset_mul in xrange(4):
            if offset_mul==2:
                continue
            start_orn = []
            for indx_tmp in xrange(3):
                start_orn.append(offset_unit*offset_mul)

            ret_val.append(qua_from_euler(start_orn))

            delta_deg = np.pi/2

            #for which_ax in xrange(3):
            for which_ax in xrange(2):
                #for mul_change_deg in xrange(3):
                for mul_change_deg in xrange(2):
                    change_deg  = mul_change_deg*delta_deg

                    new_orn     = copy.deepcopy(start_orn)
                    new_orn[which_ax] = new_orn[which_ax] + change_deg

                    ret_val.append(qua_from_euler(new_orn))

        return ret_val
    else:
        return [[0,0,0,1], [1,0,0,0], [0,1,1,1], [1,1,1,1]]


def get_scale_list(mode=0):
    if mode==0:
        return [[40], [30], [50]]
    elif mode==1:
        return [[44], [34], [54]]
    elif mode==2:
        #return [[60], [70], [80]]
        ret_val = []
        start_sc = 30
        end_sc = 90
        step_sc = 10
        for inter_sc in xrange(start_sc, end_sc, step_sc):
            ret_val.append([inter_sc])
        #return [[60], [70], [80]]
        return ret_val
    else:
        return [[40], [30], [50]]

def get_pos_list(mode=0):
    center_pos = [-10.1199,-13.1702,-22.9956]
    #start_pos = [-10.1199,10,-22.9956,0]
    #start_pos = [-11.1199,12,-20.9956,0]
    if mode==0:
        start_poses = [[-10.1199,10,-22.9956,0]]
    elif mode==2:
        start_poses = [[-10.1199,10,-22.9956,0]]
        change_pos = [-2, 2, 2]
        for indx_tmp in xrange(3):
            new_pos = copy.deepcopy(start_poses[0])
            for indx_tmp2 in xrange(3):
                if indx_tmp>0:
                    new_pos[indx_tmp2] = new_pos[indx_tmp2] + change_pos[indx_tmp2]*indx_tmp
                else:
                    new_pos[indx_tmp2] = new_pos[indx_tmp2] + change_pos[indx_tmp2]*3
            start_poses.append(new_pos)

    else:
        start_poses = [[-11.1199,12,-24.9956,0]]

    if not mode==2:
        deg_aways = [(10.0/180.0)*np.pi, -(10.0/180.0)*np.pi]
    else:
        change_deg = 10.0
        deg_aways = []
        #for indx_tmp in xrange(1, 4):
        for indx_tmp in xrange(1, 3):
            deg_aways.append( change_deg*indx_tmp/180.0*np.pi)
            deg_aways.append(-change_deg*indx_tmp/180.0*np.pi)

    which_axs = [0,2]

    ret_list = copy.deepcopy(start_poses)

    for start_pos in start_poses:
        r_now   = start_pos[1] - center_pos[1]
        for deg_away in deg_aways:
            for which_ax in which_axs:
                new_pos = copy.deepcopy(start_pos)
                new_pos[which_ax]   = new_pos[which_ax] + r_now*np.sin(deg_away)
                new_pos[1]          = center_pos[1] + r_now*np.cos(deg_away)

                #if not new_pos[2] > start_pos[2] + 1:
                #print(new_pos)
                ret_list.append(new_pos)
                #print(len(ret_list))

    return ret_list

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to generate the hdf5 data through command line')
    parser.add_argument('--pathhdf5', default = "/home/chengxuz/barrel/related_files/hdf5_files", type = str, action = 'store', help = 'Path to hdf5 folder')
    parser.add_argument('--pathexe', default = "/home/chengxuz/barrel/build_examples/Constraints/App_TestHinge", type = str, action = 'store', help = 'Path to App_ExampleBrowser')
    #parser.add_argument('--pathexe', default = "/home/chengxuz/barrel/build_examples/ExampleBrowser/App_ExampleBrowser", type = str, action = 'store', help = 'Path to App_ExampleBrowser')
    parser.add_argument('--fromcfg', default = "/home/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_", type = str, action = 'store', help = 'None means no, if the path of file sent, then get config from the file')
    parser.add_argument('--pathconfig', default = "/home/chengxuz/barrel/related_files/configs", type = str, action = 'store', help = 'Path to config folder')
    parser.add_argument('--indxsta', default = 0, type = int, action = 'store', help = 'Start index of whisker needed')
    parser.add_argument('--indxend', default = 31, type = int, action = 'store', help = 'End index of whisker needed')
    parser.add_argument('--testmode', default = 1, type = int, action = 'store', help = 'Whether run the test command or not')
    parser.add_argument('--generatemode', default = 0, type = int, action = 'store', help = 'Whether in validation mode or not')
    parser.add_argument('--checkmode', default = 0, type = int, action = 'store', help = 'Whether run the check first before running')
    parser.add_argument('--mp4flag', default = None, type = str, action = 'store', help = 'Whether generate mp4 files, if not None, will be used as mp4 name')

    parser.add_argument('--spindxsta', default = 0, type = int, action = 'store', help = 'Start index of speed')
    parser.add_argument('--spindxlen', default = 1, type = int, action = 'store', help = 'Length index of speed')
    parser.add_argument('--scindxsta', default = 0, type = int, action = 'store', help = 'Start index of scale')
    parser.add_argument('--scindxlen', default = 1, type = int, action = 'store', help = 'Length index of scale')
    parser.add_argument('--oindxsta', default = 0, type = int, action = 'store', help = 'Start index of orn')
    parser.add_argument('--oindxlen', default = 1, type = int, action = 'store', help = 'Length index of orn')
    parser.add_argument('--pindxsta', default = 0, type = int, action = 'store', help = 'Start index of pos')
    parser.add_argument('--pindxlen', default = 1, type = int, action = 'store', help = 'Length index of pos')
    parser.add_argument('--objindx', default = 0, type = int, action = 'store', help = 'Object index, 0 for duck, 1 for teddy')

    args    = parser.parse_args()

    config_dict = get_config_dict()
    config_dict = re_get_unitparams(config_dict, args.indxend - args.indxsta, args.indxsta)

    orig_config_dict = copy.deepcopy(config_dict)

    if args.fromcfg is not None:

        config_dict.pop("parameter_each")
        whisker_config_name     = []
        for curr_indx in xrange(args.indxsta, args.indxend):
            whisker_config_name.append("%s%i.cfg" % (args.fromcfg, curr_indx))
        config_dict["whisker_config_name"] = {"value":whisker_config_name, "type":"list", "type_in": "string"}

    pos_list = get_pos_list(args.generatemode)
    speed_list = get_speed_list(args.generatemode)
    orn_list = get_orn_list(args.generatemode)
    scale_list = get_scale_list(args.generatemode)

    #print(len(pos_list))
    #print(len(speed_list))
    #print(len(orn_list))
    #print(len(scale_list))

    if args.objindx==0:
        config_dict["obj_filename"]["value"] = [os.path.join(obj_path_prefix, "duck.obj")]
        hdf5_prefix = "duck"
    elif args.objindx==1:
        config_dict["obj_filename"]["value"] = [os.path.join(obj_path_prefix, "teddy.obj")]
        hdf5_prefix = "teddy"
    elif args.objindx==2:
        config_dict["obj_filename"]["value"] = [os.path.join(obj_path_prefix, "11d2b7d8c377632bd4d8765e3910f617.obj")]
        hdf5_prefix = "test"

    config_dict["add_objs"]["value"] = 1
    config_dict["time_limit"]["value"] = 11.0
    config_dict["flag_time"]["value"] = 1
    config_dict["camera_yaw"]["value"] = 183
    config_dict["camera_pitch"]["value"] = 83

    exist_num = 0
    not_exist = 0

    #for pos_array,indx in enumerate(pos_list):
    #    print(indx, pos_array)

    for indx_pos_now in xrange(args.pindxsta, args.pindxsta + args.pindxlen):
        if indx_pos_now>=len(pos_list):
            break
        config_dict["obj_pos_list"]["value"] = pos_list[indx_pos_now]
        #print(pos_list[indx_pos_now])

        for indx_scale_now in xrange(args.scindxsta, args.scindxsta + args.scindxlen):
            if indx_scale_now>=len(scale_list):
                break
            config_dict["control_len"]["value"] = scale_list[indx_scale_now]

            for indx_speed_now in xrange(args.spindxsta, args.spindxsta + args.spindxlen):
                if indx_speed_now>=len(speed_list):
                    break
                config_dict["obj_speed_list"]["value"] = speed_list[indx_speed_now]

                for indx_orn_now in xrange(args.oindxsta, args.oindxsta + args.oindxlen):
                    if indx_orn_now>=len(orn_list):
                        break
                    config_dict["obj_orn_list"]["value"] = orn_list[indx_orn_now]

                    hash_value = make_hash(config_dict)
                    hash_value = "%i_%i_%i_%i_%i" % (hash_value, indx_pos_now, indx_scale_now, indx_speed_now, indx_orn_now)
                    #config_dict["FILE_NAME"]["value"] = os.path.join(args.pathhdf5, "%s_%s.hdf5" % (hdf5_prefix, str(hash_value)))
                    config_dict["FILE_NAME"]["value"] = os.path.join(args.pathhdf5, "%s_%s.hdf5" % (hdf5_prefix, hash_value))

                    if (args.checkmode>=1):
                        if (os.path.exists(config_dict["FILE_NAME"]["value"])):
                            exist_num = exist_num + 1
                            #print("Exist:", exist_num)
                        else:
                            not_exist = not_exist + 1
                            #print("Not exist!", not_exist, config_dict["FILE_NAME"]["value"])
                        if args.checkmode==1:
                            continue

                    #now_config_fn   = os.path.join(args.pathconfig, "test_%s.cfg" % str(hash_value))
                    now_config_fn   = os.path.join(args.pathconfig, "test_%s.cfg" % hash_value)

                    # Make config files
                    #now_config_fn   = "test.cfg"

                    make_config(config_dict, now_config_fn)

                    if args.mp4flag is None:
                        if args.testmode==1:
                            cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
                        else:
                            cmd_tmp         = "%s %s"
                        cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)
                    else:
                        cmd_tmp         = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
                        cmd_str         = cmd_tmp % (args.pathexe, now_config_fn, args.mp4flag)

                    os.system(cmd_str)
                    config_dict["FILE_NAME"]["value"] = orig_config_dict["FILE_NAME"]["value"]
    if args.checkmode>=1:
        print(exist_num)
