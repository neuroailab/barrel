#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_hdf5_%j.out

source activate env_torch
python cmd_hdf5.py --pathhdf5 /om/user/chengxuz/barrel/hdf5s --pathexe /om/user/chengxuz/barrel/example_build/Constraints/App_TestHinge --fromcfg /om/user/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /om/user/chengxuz/barrel/configs --objindx ${1} --spindxlen 3 --scindxlen 3 --oindxsta ${2} --pindxsta ${3}
