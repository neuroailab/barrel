#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 1
#SBATCH --mem=9000
#SBATCH --output=/mnt/fs0/chengxuz/slurm_out_all/part_dismat_%j.out

python compute_part_dismat.py --xstart ${1} --xlen ${2} --ystart ${3} --ylen ${4} --savesuffix concat.pkl
