#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 2
#SBATCH --output=/mnt/fs0/chengxuz/slurm_out_all/scenenet_normal_totfrecs_%j.out

python combine_tfrecs.py --loaddir /mnt/fs1/Dataset/scenenet/${1} --savedir /mnt/fs1/Dataset/scenenet_combine/${1} --indxsta ${2} --indxlen ${3}
