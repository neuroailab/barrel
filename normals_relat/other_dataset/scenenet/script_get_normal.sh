#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 6
#SBATCH --output=/mnt/fs0/chengxuz/slurm_out_all/scenenet_normal_gettfrecs_%j.out

python get_normal_again_from_tfrecs.py --indxsta ${1} --indxlen ${2}
