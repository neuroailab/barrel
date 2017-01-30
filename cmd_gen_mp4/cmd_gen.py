import argparse
import os
from itertools import *
import copy
import multiprocessing
import numpy as np
from get_ratMap import get_wholeS
import copy
import subprocess
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials

'''
The script to run the whisker thing and generate the mp4s through command line
'''

args        = []
all_items   = []
config_dict = {}
para_search = {}
nu_rela_key = ['camera_dist', 'time_limit']
y_len_link = 1.04

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

def build_array(num_whskr):
    x_pos_base      = []
    y_pos_base      = []
    z_pos_base      = []
    const_numLinks  = []
    qua_list        = []
    yaw_y_base      = []
    pitch_x_base    = []
    roll_z_base     = []

    x_pos_st        = -0.4
    x_pos_step      = 10
    y_pos_va        = 7
    z_pos_st        = 0
    z_pos_step      = 10
    const_num_l     = 25

    qua_st          = -0.1
    yaw_y_base_st   = 0.5
    pitch_x_base_st = 0.3
    roll_z_base_st  = 0.6

    S   = get_wholeS()

    for indx_w in xrange(num_whskr):
        x_pos_base.append(S.C_baseZ[indx_w])
        y_pos_base.append(S.C_baseY[indx_w])
        z_pos_base.append(S.C_baseX[indx_w])

        const_numLinks.append(np.ceil(S.C_s[indx_w]/(y_len_link*2)))
        qua_list.append(-S.C_a[indx_w])

        yaw_y_base.append(S.C_phi[indx_w])
        pitch_x_base.append(S.C_zeta[indx_w])
        roll_z_base.append(S.C_theta[indx_w])

    return {'x':x_pos_base, 'y':y_pos_base, 'z':z_pos_base, 'c':const_numLinks, 
            'yaw':yaw_y_base, 'pitch':pitch_x_base, 'roll':roll_z_base, 'qua':qua_list}

def make_hash(o):

    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """

    if isinstance(o, (set, tuple, list)):

        return tuple([make_hash(e) for e in o])        

    elif not isinstance(o, dict):

        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))

array_dict      = build_array(1)

every_spring_value = []
inter_spring_value = []
for indx_spring in xrange(3, 30):
    every_spring_value.append(indx_spring)
    inter_spring_value.append(1)


config_dict     = {"x_len_link":{"value":0.53, "help":"Size x of cubes", "type":"float"}, 
        "y_len_link":{"value":y_len_link, "help":"Size y of cubes", "type":"float"},
        "z_len_link":{"value":0.3, "help":"Size z of cubes", "type":"float"}, 
        "basic_str":{"value":2508, "help":"Minimal strength of hinge's recover force", "type":"float"}, 
        "x_pos_base":{"value":array_dict['x'], "help":"Position x of base", "type":"list", "type_in":"float"},
        "y_pos_base":{"value":array_dict['y'], "help":"Position y of base", "type":"list", "type_in":"float"},
        "z_pos_base":{"value":array_dict['z'], "help":"Position z of base", "type":"list", "type_in":"float"},
        "const_numLinks":{"value":array_dict['c'], "help":"Number of units", "type":"list", "type_in":"int"},
        "yaw_y_base":{"value":array_dict['yaw'], "help":"Yaw of base", "type":"list", "type_in":"float"},
        "pitch_x_base":{"value":array_dict['pitch'], "help":"Pitch of base", "type":"list", "type_in":"float"},
        "roll_z_base":{"value":array_dict['roll'], "help":"Roll of base", "type":"list", "type_in":"float"},
        "qua_a_list":{"value":array_dict['qua'], "help":"Quadratic Coefficient", "type":"list", "type_in":"float"},
        "inter_spring":{"value":inter_spring_value, "help":"Number of units between two strings", "type":"list", "type_in": "int"}, 
        "every_spring":{"value":every_spring_value, "help":"Number of units between one strings", "type":"list", "type_in": "int"},
        #"linear_damp":{"value":0.7, "help":"Control the linear damp ratio", "type":"float"},
        #"linear_damp":{"value":0.997, "help":"Control the linear damp ratio", "type":"float"},
        "linear_damp":{"value":0.95, "help":"Control the linear damp ratio", "type":"float"},
        #"ang_damp":{"value":0.7, "help":"Control the angle damp ratio", "type":"float"},
        #"ang_damp":{"value":0.18, "help":"Control the angle damp ratio", "type":"float"},
        "ang_damp":{"value":0.54, "help":"Control the angle damp ratio", "type":"float"},
        "time_leap":{"value":1.0/240.0, "help":"Time unit for simulation", "type":"float"},
        "equi_angle":{"value":0, "help":"Control the angle of balance for hinges", "type":"float"}, 
        "spring_stiffness":{"value":3233, "help":"Stiffness of spring", "type":"float"}, 
        "spring_stfperunit":{"value":2117, "help":"Stiffness of spring per unit", "type":"float"}, 
        "camera_dist":{"value":40, "help":"Distance of camera", "type":"float", "dict_nu":{5: 20, 15:45, 25:70}}, 
        "spring_offset":{"value":0, "help":"String offset for balance state", "type":"float"}, 
        "time_limit":{"value":50.0/4, "help":"Time limit for recording", "type":"float", "dict_nu": {5: 20.0/4, 15: 35.0/4, 25:50.0/4}}, 
        "initial_str":{"value":10000, "help":"Initial strength of force applied", "type":"float"}, 
        "max_str":{"value":10000, "help":"Max strength of force applied", "type":"float"}, 
        "initial_stime":{"value":2.1/8, "help":"Initial time to apply force", "type":"float"}, 
        "angl_ban_limit":{"value":0.5, "help":"While flag_time is 2, used for angular velocities of rigid bodys to judge whether stop", "type":"float"}, 
        "velo_ban_limit":{"value":0.5, "help":"While flag_time is 2, used for linear velocities of rigid bodys to judge whether stop", "type":"float"}, 
        "state_ban_limit":{"value":0.5, "help":"While flag_time is 2, used for angle states of hinges to judge whether stop", "type":"float"}, 
        "force_limit":{"value":40, "help":"While flag_time is 2, used for force states of hinges to judge whether stop", "type":"float"}, 
        "torque_limit":{"value":120, "help":"While flag_time is 2, used for torque states of hinges to judge whether stop", "type":"float"}, 
        "hinge_mode":{"value":0, "help":"Whether use hinges rather than springs for connections of two units", "type":"int"},
        "test_mode":{"value":0, "help":"Whether enter test mode for some temp test codes, default is 0", "type":"int"},
        "force_mode":{"value":2, "help":"Force mode to apply at the beginning, default is 0", "type":"int"},
        "flag_time":{"value":2, "help":"Whether open time limit", "type":"int"}}

def get_value(kwargs, pathconfig ="/scratch/users/chengxuz/barrel/barrel_relat_files/configs", pathexe ="/scratch/users/chengxuz/barrel/examples_build/Constraints/App_TestHinge",  
        coe_curr_dis = 1.0/40.0, coe_min_dis = 1.0, coe_all_time = 20.0):

    for key, value in kwargs.iteritems():
        if key in config_dict:
            config_dict[key]['value'] = value

    inner_loop = {0: {'force_mode': 0, "initial_str": 30000}, 1: {'force_mode': 1, "initial_str": 10000}, 2: {'force_mode': 2, "initial_str": 10000}}

    all_ret_val = 0
    for key, value in inner_loop.iteritems():
        for key_i, value_i in value.iteritems():
            if key_i in config_dict:
                config_dict[key_i]['value'] = value_i

        hash_value = make_hash(config_dict)
        #print(hash_value)
        #print(pathconfig)
        now_config_fn   = os.path.join(pathconfig, "test_%s.cfg" % str(hash_value))

        make_config(config_dict, now_config_fn)

        tmp_outputs = subprocess.check_output([pathexe, now_config_fn])
        tmp_splits = tmp_outputs.split('\n')
        curr_dis = float(tmp_splits[1].split(':')[1])
        min_dis = float(tmp_splits[2].split(':')[1])
        all_time = float(tmp_splits[3].split(':')[1])
        retval = coe_curr_dis*curr_dis + coe_min_dis*min_dis + coe_all_time*all_time

        all_ret_val = all_ret_val + retval

    return all_ret_val


_default_prior_weight = 1.0

# -- suggest best of this many draws on every iteration
_default_n_EI_candidates = 24

# -- gamma * sqrt(n_trials) is fraction of to use as good
_default_gamma = 0.25

_default_n_startup_jobs = 100

def my_suggest(new_ids, domain, trials, seed,
            prior_weight=_default_prior_weight,
            n_startup_jobs=_default_n_startup_jobs,
            n_EI_candidates=_default_n_EI_candidates,
            gamma=_default_gamma):
    return tpe.suggest(new_ids, domain, trials, seed,
            prior_weight, n_startup_jobs, n_EI_candidates, gamma)

def do_hyperopt(eval_num, use_mongo = False, portn = 23333, db_name = "test_db", exp_name = "exp1"):

    if (use_mongo):
        trials = MongoTrials('mongo://localhost:%i/%s/jobs' % (portn, db_name), exp_key=exp_name)
    else:
        trials = Trials()

    best = fmin(fn=get_value, 
        space=hp.choice('a', [
            {'basic_str': hp.uniform('b_s', 1000, 9000), 'linear_damp':hp.uniform('l_d', 0, 1), 'ang_damp':hp.uniform('a_d', 0, 1),
                'spring_stiffness': hp.uniform('s_s', 100, 5000), 'spring_stfperunit':hp.uniform('s_sp', 1000, 9000)},
            ]),
        algo=my_suggest,
        trials=trials,
        max_evals=eval_num)
    print best

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

    if args.testmode==0:
        para_search     = {"basic_str":{"range":[1000, 3000, 5000, 7000], "short":"ba"}, 
                "const_numLinks":{"range":[5, 15, 25], "short":"nu"},
                #"const_numLinks":{"range":[25], "short":"nu"},
                "damp":{"range":[0.1, 0.5, 0.9], "key_value": ["linear_damp", "ang_damp"], "short":"dp"},
                "inter_spring":{"range":[1, 3, 5, 7], "short":"is"},
                "every_spring":{"range":[2, 3, 5, 7, 9, 11, 13, 17, 21], "short":"es"},
                "spring_stiffness":{"range":[300, 500, 700, 900], "short":"ss"}
                }

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


        nu_args     = range(int(np.ceil(len(all_items)*1.0/args.mapn)))
        #nu_args     = range(2)
        pool = multiprocessing.Pool(processes=args.nproc)
        r = pool.map_async(run_it, nu_args)
        r.get()
        #print('Done!')
        #run_it(0)
    elif args.testmode==1:
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
    else:
        now_config_fn   = "test.cfg"

        print(make_hash(config_dict))
        make_config(config_dict, now_config_fn)
        cmd_tmp         = "%s %s"
        cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)

        os.system(cmd_str)
