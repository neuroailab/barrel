import argparse
import os

def make_config(config_dict, config_filename):
    fout        = open(config_filename, 'w')
    line_tmp    = {"float":"%s=%f\n", "int":"%s=%i\n"}
    line_com    = "#%s\n"

    for key_value in config_dict:
        tmp_key     = config_dict[key_value]
        fout.write(line_com % tmp_key["help"])
        fout.write(line_tmp[tmp_key["type"]] % (key_value, tmp_key["value"]))
        fout.write("\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='The script to generate the mp4s through command line')
    parser.add_argument('--nproc', default = 4, type = int, action = 'store')
    parser.add_argument('--pathconfig', default = "/home/chengxuz/barrel/barrel/bullet_demos_extracted/configs", type = str, action = 'store')
    parser.add_argument('--pathmp4', default = "/home/chengxuz/barrel/barrel/cmd_gen_mp4/generated_mp4s", type = str, action = 'store')
    parser.add_argument('--pathexe', default = "/home/chengxuz/barrel/build_examples/ExampleBrowser/App_ExampleBrowser", type = str, action = 'store')

    args    = parser.parse_args()
    #print(args.nproc)
    config_dict     = {"x_len_link":{"value":0.53, "help":"Size x of cubes", "type":"float"}, 
            "y_len_link":{"value":2.08, "help":"Size y of cubes", "type":"float"},
            "z_len_link":{"value":0.3, "help":"Size z of cubes", "type":"float"}, 
            "basic_str":{"value":3000, "help":"Minimal strength of hinge's recover force", "type":"float"}, 
            "const_numLinks":{"value":15, "help":"Number of units", "type":"int"},
            "linear_damp":{"value":0.17, "help":"Control the linear damp ratio", "type":"float"},
            "ang_damp":{"value":0.19, "help":"Control the angle damp ratio", "type":"float"},
            "time_leap":{"value":1.0/240.0, "help":"Time unit for simulation", "type":"float"},
            "equi_angle":{"value":0, "help":"Control the angle of balance for hinges", "type":"float"}, 
            "inter_spring":{"value":5, "help":"Number of units between two strings", "type":"int"}, 
            "every_spring":{"value":3, "help":"Number of units between one strings", "type":"int"},
            "spring_stiffness":{"value":520, "help":"Stiffness of spring", "type":"float"}, 
            "camera_dist":{"value":45, "help":"Distance of camera", "type":"float"}, 
            "spring_offset":{"value":0, "help":"String offset for balance state", "type":"float"}, 
            "time_limit":{"value":30.0/4, "help":"Time limit for recording", "type":"float"}, 
            "initial_str":{"value":30000, "help":"Initial strength of force applied", "type":"float"}, 
            "initial_stime":{"value":0.1/8, "help":"Initial time to apply force", "type":"float"}, 
            "limit_softness":{"value":0.9, "help":"Softness of the hinge limit", "type":"float"}, 
            "limit_bias":{"value":0.3, "help":"Bias of the hinge limit", "type":"float"}, 
            "limit_relax":{"value":1, "help":"Relax of the hinge limit", "type":"float"}, 
            "limit_low":{"value":-2, "help":"Low bound of the hinge limit", "type":"float"}, 
            "limit_up":{"value":0, "help":"Up bound of the hinge limit", "type":"float"}, 
            "angl_ban_limit":{"value":1.5, "help":"While flag_time is 2, used for angular velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "velo_ban_limit":{"value":1.5, "help":"While flag_time is 2, used for linear velocities of rigid bodys to judge whether stop", "type":"float"}, 
            "state_ban_limit":{"value":1, "help":"While flag_time is 2, used for angle states of hinges to judge whether stop", "type":"float"}, 
            "force_limit":{"value":100, "help":"While flag_time is 2, used for force states of hinges to judge whether stop", "type":"float"}, 
            "torque_limit":{"value":200, "help":"While flag_time is 2, used for torque states of hinges to judge whether stop", "type":"float"}, 
            "initial_poi":{"value":14, "help":"Unit to apply the force", "type":"int"}, 
            "flag_time":{"value":1, "help":"Whether open time limit", "type":"int"}}

    # Make config files
    now_config_fn   = "test.cfg"
    now_mp4_fn      = "test.mp4"
    cmd_tmp         = "%s --config_filename=%s --mp4=%s --start_demo_name=TestHingeTorque"
    #cmd_tmp         = "%s --config_filename=%s --start_demo_name=TestHingeTorque"
    make_config(config_dict, now_config_fn)
    cmd_str         = cmd_tmp % (args.pathexe, now_config_fn, now_mp4_fn)
    #cmd_str         = cmd_tmp % (args.pathexe, now_config_fn)
    os.system(cmd_str)
