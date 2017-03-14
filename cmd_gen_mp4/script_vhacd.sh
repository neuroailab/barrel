#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -c 5
#SBATCH --mem=10000
#SBATCH --output=/om/user/chengxuz/slurm_out_all/barrel_vhacd_%j.out

source activate env_torch_2
ssh -f -N -L 22334:localhost:22334 chengxuz@171.64.40.90
python cmd_vhacd.py --startIndx ${1} --lenIndx ${2}
