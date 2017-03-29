#!/bin/bash
#SBATCH --time=6-00:00:00
#SBATCH -c 6
#SBATCH --output=/om/user/chengxuz/slurm_out_all/scenenet_normal_totfrecs_%j.out

source activate env_torch_2

python get_normal_tfrecs.py --path /om/user/chengxuz/Data/one_world_dataset/split_scenenet/train_tfrecords/${1}/normal --hdf5 /om/user/chengxuz/Data/one_world_dataset/split_scenenet/hdf5s/train_${1}.hdf5 --seed ${1}
