#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -c 1
#SBATCH --mem=3000
#SBATCH --output=/om/user/chengxuz/slurm_out_all/part_RDM_%j.out

python compute_part_RDM.py --xstart ${1} --xlen ${2} --ystart ${3} --ylen ${4} --hdf5path /om/user/chengxuz/Data/spa_temp_resps/conv${5}.hdf5 --keypat conv${5}_%i --savesuffix conv${5} --savefolder /om/user/chengxuz/Data/part_RDMs
