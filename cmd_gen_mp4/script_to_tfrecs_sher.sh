#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --output=/scratch/users/chengxuz/slurm_output/barrel_tfrecs_%j.out

module load tensorflow/0.12.1
module load anaconda/anaconda.4.2.0.python2.7
python cmd_to_tfrecords.py --objsta ${1} --objlen ${2} --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/tfrecords --infodir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_info --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --seedbas 10000
