import argparse
import os
from itertools import *
import copy
import multiprocessing
import numpy as np

args        = []
all_items   = []
config_dict = {}
para_search = {}
nu_rela_key = ['camera_dist', 'time_limit']

def make_config(config_dict, config_filename):
    fout        = open(config_filename, 'w')
    line_tmp    = {"float":"%s=%f\n", "int":"%s=%i\n"}
    line_com    = "#%s\n"

    for key_value in config_dict:
        tmp_key     = config_dict[key_value]
        fout.write(line_com % tmp_key["help"])
        if not tmp_key["type"]=="list":
            fout.write(line_tmp[tmp_key["type"]] % (key_value, tmp_key["value"]))
        else:
            #fout.write(line_tmp[tmp_key["type"]] % (key_value, str(tmp_key["value"])[1:-1].replace(',', '') ))
            #fout.write(line_tmp[tmp_key["type"]] % (key_value, str(tmp_key["value"])[1:-1] ))
            for value_list in tmp_key["value"]:
                fout.write(line_tmp[tmp_key["type_in"]] % (key_value, value_list))
        fout.write("\n")

def my_product(dicts):
    return (dict(izip(dicts, x)) for x in product(*[x['range'] for x in dicts.itervalues()]))

def run_it(ind):
    global all_items 
    global args
    global config_dict
    global para_search

    cmd_tmp1        = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
    cmd_tmp2        = "%s --config_filename=%s --start_demo_name=TestHingeTorque"

    start_indx  = min(ind * args.mapn, len(all_items))
    end_indx    = min((ind+1)*args.mapn, len(all_items))
    curr_items  = all_items[start_indx: end_indx]

    our_config  = copy.deepcopy(config_dict)
    
    for item in curr_items:

        now_file_name   = ""

        for key_value in item:
            if not key_value=='damp':
                our_config[key_value]["value"]  = item[key_value]
            else:
                for sub_key in para_search[key_value]["key_value"]:
                    our_config[sub_key]["value"]  = item[key_value]
            now_file_name   = now_file_name + para_search[key_value]["short"] + str(para_search[key_value]["range"].index(item[key_value]))

        now_nu  = item["const_numLinks"]
        our_config['initial_poi']["value"]  = now_nu - 1
        #our_config['camera_dist']["value"]  = dist_dict[now_nu]
        for tmp_key in nu_rela_key:
            our_config[tmp_key]["value"]  = our_config[tmp_key]["dict_nu"][now_nu]

        # Make config files
        now_config_fn   = os.path.join(args.pathconfig, now_file_name + ".cfg")
        now_mp4_fn      = os.path.join(args.pathmp4, now_file_name + ".mp4")

        make_config(our_config, now_config_fn)
        if args.mp4flag==1:
            cmd_tmp         = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn, now_mp4_fn)
        else:
            cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)

        os.system(cmd_str)

def build_array(x, y):
    x_pos_base      = []
    y_pos_base      = []
    z_pos_base      = []
    const_numLinks  = []

    x_pos_st        = -0.4
    x_pos_step      = 10
    y_pos_va        = 7
    z_pos_st        = 0
    z_pos_step      = 10
    const_num_l     = 25
    #const_num_l     = 4
    #const_num_l     = 2

    for indx_x in range(x):
        for indx_y in range(y):
            x_pos_base.append(x_pos_st + indx_x*x_pos_step)
            z_pos_base.append(z_pos_st + indx_y*z_pos_step)
            y_pos_base.append(y_pos_va)
            const_numLinks.append(const_num_l)

    return {'x':x_pos_base, 'y':y_pos_base, 'z':z_pos_base, 'c':const_numLinks}
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to generate the mp4s through command line')
    parser.add_argument('--nproc', default = 4, type = int, action = 'store', help = 'Number of processes')
    parser.add_argument('--pathconfig', default = "/home/chengxuz/barrel/barrel/bullet_demos_extracted/configs", type = str, action = 'store', help = 'Path to config folder')
    parser.add_argument('--pathmp4', default = "/home/chengxuz/barrel/barrel/cmd_gen_mp4/generated_mp4s", type = str, action = 'store', help = 'Path to mp4 folder')
    parser.add_argument('--pathexe', default = "/home/chengxuz/barrel/build_examples/ExampleBrowser/App_ExampleBrowser", type = str, action = 'store', help = 'Path to App_ExampleBrowser')
    parser.add_argument('--mp4flag', default = 1, type = int, action = 'store', help = 'Whether generate mp4 files')
    parser.add_argument('--mapn', default = 30, type = int, action = 'store', help = 'Number of items in each processes')
    parser.add_argument('--testmode', default = 0, type = int, action = 'store', help = 'Whether run the test command or not')

    args    = parser.parse_args()
    #print(args.nproc)

    #array_dict      = build_array(3,3)
    array_dict      = build_array(1,1)

    config_dict     = {"x_len_link":{"value":0.53, "help":"Size x of cubes", "type":"float"}, 
            "y_len_link":{"value":2.08, "help":"Size y of cubes", "type":"float"},
            "z_len_link":{"value":0.3, "help":"Size z of cubes", "type":"float"}, 
            #"basic_str":{"value":3000, "help":"Minimal strength of hinge's recover force", "type":"float"}, 
            "basic_str":{"value":100, "help":"Minimal strength of hinge's recover force", "type":"float"}, 

            "x_pos_base":{"value":array_dict['x'], "help":"Position x of base", "type":"list", "type_in":"float"},
            "y_pos_base":{"value":array_dict['y'], "help":"Position y of base", "type":"list", "type_in":"float"},
            "z_pos_base":{"value":array_dict['z'], "help":"Position z of base", "type":"list", "type_in":"float"},
            "const_numLinks":{"value":array_dict['c'], "help":"Number of units", "type":"list", "type_in":"int"},
            #"inter_spring":{"value":(1, 3, 3, 3), "help":"Number of units between two strings", "type":"list", "type_in": "int"}, 
            #"every_spring":{"value":(3, 5, 7, 9), "help":"Number of units between one strings", "type":"list", "type_in": "int"},
            "inter_spring":{"value":(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), "help":"Number of units between two strings", "type":"list", "type_in": "int"}, 
            "every_spring":{"value":(3, 5, 7, 9, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), "help":"Number of units between one strings", "type":"list", "type_in": "int"},

            "linear_damp":{"value":0.5, "help":"Control the linear damp ratio", "type":"float"},
            "ang_damp":{"value":0.5, "help":"Control the angle damp ratio", "type":"float"},
            #"linear_damp":{"value":0.1, "help":"Control the linear damp ratio", "type":"float"},
            #"ang_damp":{"value":0.1, "help":"Control the angle damp ratio", "type":"float"},
            "time_leap":{"value":1.0/240.0, "help":"Time unit for simulation", "type":"float"},
            "equi_angle":{"value":0, "help":"Control the angle of balance for hinges", "type":"float"}, 
            #"equi_angle":{"value":-0.05, "help":"Control the angle of balance for hinges", "type":"float"}, 
            "spring_stiffness":{"value":500, "help":"Stiffness of spring", "type":"float"}, 
            #"spring_stiffness":{"value":100, "help":"Stiffness of spring", "type":"float"}, 
            #"camera_dist":{"value":90, "help":"Distance of camera", "type":"float", "dict_nu":{5: 20, 15:45, 25:70}}, 
            "camera_dist":{"value":20, "help":"Distance of camera", "type":"float", "dict_nu":{5: 20, 15:45, 25:70}}, 
            "spring_offset":{"value":0, "help":"String offset for balance state", "type":"float"}, 
            "time_limit":{"value":50.0/4, "help":"Time limit for recording", "type":"float", "dict_nu": {5: 20.0/4, 15: 35.0/4, 25:50.0/4}}, 
            #"initial_str":{"value":50000, "help":"Initial strength of force applied", "type":"float"}, 
            "initial_str":{"value":150000, "help":"Initial strength of force applied", "type":"float"}, 
            "initial_stime":{"value":0.1/8, "help":"Initial time to apply force", "type":"float"}, 
            "limit_softness":{"value":0.9, "help":"Softness of the hinge limit", "type":"float"}, 
            "limit_bias":{"value":0.3, "help":"Bias of the hinge limit", "type":"float"}, 
            "limit_relax":{"value":1, "help":"Relax of the hinge limit", "type":"float"}, 
            #"limit_low":{"value":-2, "help":"Low bound of the hinge limit", "type":"float"}, 
            #"limit_low":{"value":0.1, "help":"Low bound of the hinge limit", "type":"float"}, 
            "limit_low":{"value":0, "help":"Low bound of the hinge limit", "type":"float"}, 
            #"limit_up":{"value":-0.1, "help":"Up bound of the hinge limit", "type":"float"}, 
            "limit_up":{"value":2, "help":"Up bound of the hinge limit", "type":"float"}, 
            #"limit_up":{"value":2, "help":"Up bound of the hinge limit", "type":"float"}, 
            "angl_ban_limit":{"value":1.5, "help":"While flag_time is 2, used for angular velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "velo_ban_limit":{"value":1.5, "help":"While flag_time is 2, used for linear velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "state_ban_limit":{"value":1, "help":"While flag_time is 2, used for angle states of hinges to judge whether stop", "type":"float"}, 
            "force_limit":{"value":100, "help":"While flag_time is 2, used for force states of hinges to judge whether stop", "type":"float"}, 
            "torque_limit":{"value":200, "help":"While flag_time is 2, used for torque states of hinges to judge whether stop", "type":"float"}, 
            "initial_poi":{"value":24, "help":"Unit to apply the force", "type":"int"}, 
            "hinge_mode":{"value":0, "help":"Whether use hinges rather than springs for connections of two units", "type":"int"},
            "test_mode":{"value":1, "help":"Whether enter test mode for some temp test codes, default is 0", "type":"int"},
            "flag_time":{"value":0, "help":"Whether open time limit", "type":"int"}}
    '''
    config_dict     = {"x_len_link":{"value":0.53, "help":"Size x of cubes", "type":"float"}, 
            "y_len_link":{"value":2.08, "help":"Size y of cubes", "type":"float"},
            "z_len_link":{"value":0.3, "help":"Size z of cubes", "type":"float"}, 
            #"basic_str":{"value":9000000, "help":"Minimal strength of hinge's recover force", "type":"float"}, 
            "basic_str":{"value":3000, "help":"Minimal strength of hinge's recover force", "type":"float"}, 
            "const_numLinks":{"value":25, "help":"Number of units", "type":"int"},
            "linear_damp":{"value":0.5, "help":"Control the linear damp ratio", "type":"float"},
            #"linear_damp":{"value":0, "help":"Control the linear damp ratio", "type":"float"},
            "ang_damp":{"value":0.5, "help":"Control the angle damp ratio", "type":"float"},
            #"ang_damp":{"value":0, "help":"Control the angle damp ratio", "type":"float"},
            "time_leap":{"value":1.0/240.0, "help":"Time unit for simulation", "type":"float"},
            "equi_angle":{"value":0, "help":"Control the angle of balance for hinges", "type":"float"}, 
            "inter_spring":{"value":(1, 3, 3, 3), "help":"Number of units between two strings", "type":"list", "type_in": "int"}, 
            "every_spring":{"value":(3, 5, 7, 9), "help":"Number of units between one strings", "type":"list", "type_in": "int"},
            "spring_stiffness":{"value":1000, "help":"Stiffness of spring", "type":"float"}, 
            "camera_dist":{"value":70, "help":"Distance of camera", "type":"float", "dict_nu":{5: 20, 15:45, 25:70}}, 
            "spring_offset":{"value":0, "help":"String offset for balance state", "type":"float"}, 
            "time_limit":{"value":50.0/4, "help":"Time limit for recording", "type":"float", "dict_nu": {5: 20.0/4, 15: 35.0/4, 25:50.0/4}}, 
            "initial_str":{"value":50000, "help":"Initial strength of force applied", "type":"float"}, 
            "initial_stime":{"value":0.1/8, "help":"Initial time to apply force", "type":"float"}, 
            "limit_softness":{"value":0.9, "help":"Softness of the hinge limit", "type":"float"}, 
            "limit_bias":{"value":0.3, "help":"Bias of the hinge limit", "type":"float"}, 
            "limit_relax":{"value":1, "help":"Relax of the hinge limit", "type":"float"}, 
            "limit_low":{"value":0.05, "help":"Low bound of the hinge limit", "type":"float"}, 
            "limit_up":{"value":2, "help":"Up bound of the hinge limit", "type":"float"}, 
            "angl_ban_limit":{"value":1.5, "help":"While flag_time is 2, used for angular velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "velo_ban_limit":{"value":1.5, "help":"While flag_time is 2, used for linear velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "state_ban_limit":{"value":1, "help":"While flag_time is 2, used for angle states of hinges to judge whether stop", "type":"float"}, 
            "force_limit":{"value":100, "help":"While flag_time is 2, used for force states of hinges to judge whether stop", "type":"float"}, 
            "torque_limit":{"value":200, "help":"While flag_time is 2, used for torque states of hinges to judge whether stop", "type":"float"}, 
            "initial_poi":{"value":24, "help":"Unit to apply the force", "type":"int"}, 
            "hinge_mode":{"value":0, "help":"Whether use hinges rather than springs for connections of two units", "type":"int"},
            #"hinge_mode":{"value":1, "help":"Whether use hinges rather than springs for connections of two units", "type":"int"},
            "flag_time":{"value":0, "help":"Whether open time limit", "type":"int"}}
    '''

    if args.testmode==0:
        para_search     = {"basic_str":{"range":[1000, 3000, 5000, 7000], "short":"ba"}, 
                "const_numLinks":{"range":[5, 15, 25], "short":"nu"},
                #"const_numLinks":{"range":[25], "short":"nu"},
                "damp":{"range":[0.1, 0.5, 0.9], "key_value": ["linear_damp", "ang_damp"], "short":"dp"},
                "inter_spring":{"range":[1, 3, 5, 7], "short":"is"},
                "every_spring":{"range":[2, 3, 5, 7, 9, 11, 13, 17, 21], "short":"es"},
                "spring_stiffness":{"range":[300, 500, 700, 900], "short":"ss"}
                }

        #print(len(list(my_product(para_search))))

        for check_item in my_product(para_search):

            right_flag      = 1

            for key_value in check_item:
                if check_item[key_value] not in para_search[key_value]['range']:
                    right_flag  = 0

            if check_item['every_spring'] > check_item["const_numLinks"]:
                right_flag      = 0

            if check_item['inter_spring'] > check_item["const_numLinks"]:
                right_flag      = 0

            if right_flag==1:
                all_items.append(check_item)

        #print(len(all_items), all_items[0])


        nu_args     = range(int(np.ceil(len(all_items)*1.0/args.mapn)))
        #nu_args     = range(2)
        pool = multiprocessing.Pool(processes=args.nproc)
        r = pool.map_async(run_it, nu_args)
        r.get()
        #print('Done!')
        #run_it(0)
    else:
        # Make config files
        now_config_fn   = "test.cfg"
        now_mp4_fn      = "test.mp4"

        make_config(config_dict, now_config_fn)
        if args.mp4flag==1:
            cmd_tmp         = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn, now_mp4_fn)
        else:
            cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
            cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)

        os.system(cmd_str)
