#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train.sh 1
sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train_tfr.sh 1 trainval1_tfr 27017
