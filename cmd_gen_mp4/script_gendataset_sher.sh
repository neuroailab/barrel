#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --output=/scratch/users/chengxuz/slurm_output/barrel_gendata_%j.out

#python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000

# Change to write info files
python cmd_dataset.py --objsta ${1} --objlen ${2} --bigsamnum ${3} --pathexe /scratch/users/chengxuz/barrel/examples_build_2/Constraints/App_TestHinge --fromcfg /scratch/users/chengxuz/barrel/barrel/cmd_gen_mp4/opt_results/para_ --pathconfig /scratch/users/chengxuz/barrel/barrel_relat_files/configs --savedir /scratch/users/chengxuz/barrel/barrel_relat_files/dataset/raw_hdf5 --loaddir /scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd --seedbas 10000 --checkmode 2
