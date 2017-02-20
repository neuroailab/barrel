import argparse
import os
from itertools import *
import copy
from cmd_gen import *
import numpy as np

def get_speed_list():
    #return [[0,-12,0], [0, -10, 0], [0, -8, 0]]
    return [[0,-12.5,0], [0, -10.5, 0], [0, -8.5, 0]]

def get_orn_list():
    return [[0,0,0,1], [1,0,0,0], [0,1,1,1], [1,1,1,1]]

def get_scale_list():
    #return [[40], [30], [50]]
    return [[44], [34], [54]]

def get_pos_list():
    center_pos = [-10.1199,-13.1702,-22.9956]
    #start_pos = [-10.1199,10,-22.9956,0]
    start_pos = [-11.1199,12,-20.9956,0]

    deg_aways = [(10.0/180.0)*np.pi, -(10.0/180.0)*np.pi]
    which_axs = [0,2]

    ret_list = [start_pos]

    r_now   = start_pos[1] - center_pos[1]
    for deg_away in deg_aways:
        for which_ax in which_axs:
            new_pos = copy.deepcopy(start_pos)
            new_pos[which_ax]   = new_pos[which_ax] + r_now*np.sin(deg_away)
            new_pos[1]          = center_pos[1] + r_now*np.cos(deg_away)

            if not new_pos[2] > start_pos[2] + 1:
                ret_list.append(new_pos)

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

    pos_list = get_pos_list()
    speed_list = get_speed_list()
    orn_list = get_orn_list()
    scale_list = get_scale_list()

    #print(pos_list)

    if args.objindx==0:
        config_dict["obj_filename"]["value"] = [os.path.join(obj_path_prefix, "duck.obj")]
        hdf5_prefix = "duck"
    elif args.objindx==1:
        config_dict["obj_filename"]["value"] = [os.path.join(obj_path_prefix, "teddy.obj")]
        hdf5_prefix = "teddy"

    for indx_pos_now in xrange(args.pindxsta, args.pindxsta + args.pindxlen):
        config_dict["obj_pos_list"]["value"] = pos_list[indx_pos_now]

        for indx_scale_now in xrange(args.scindxsta, args.scindxsta + args.scindxlen):
            config_dict["control_len"]["value"] = scale_list[indx_scale_now]

            for indx_speed_now in xrange(args.spindxsta, args.spindxsta + args.spindxlen):
                config_dict["obj_speed_list"]["value"] = speed_list[indx_speed_now]

                for indx_orn_now in xrange(args.oindxsta, args.oindxsta + args.oindxlen):
                    config_dict["obj_orn_list"]["value"] = orn_list[indx_orn_now]

                    hash_value = make_hash(config_dict)
                    config_dict["FILE_NAME"]["value"] = os.path.join(args.pathhdf5, "%s_%s.hdf5" % (hdf5_prefix, str(hash_value)))

                    now_config_fn   = os.path.join(args.pathconfig, "test_%s.cfg" % str(hash_value))

                    # Make config files
                    #now_config_fn   = "test.cfg"

                    make_config(config_dict, now_config_fn)

                    if args.testmode==1:
                        cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
                    else:
                        cmd_tmp         = "%s %s"

                    cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)

                    os.system(cmd_str)
