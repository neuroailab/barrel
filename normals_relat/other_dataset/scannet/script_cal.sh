#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 5
#SBATCH --mem=8000
#SBATCH --output=/om/user/chengxuz/slurm_out_all/scannet_tfrecs_%j.out

source activate env_torch_2
python meta_cal.py --indxsta ${1} --indxlen ${2}
