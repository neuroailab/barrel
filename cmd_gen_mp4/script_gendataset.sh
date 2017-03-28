#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 1
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_gendataset_%j.out

source activate env_torch_2
#ssh -f -N -L 22334:localhost:22334 chengxuz@dicarlo5.mit.edu
#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3}

# Change to write the information
python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --checkmode 2
