#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train.sh 1
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train_tfr.sh 1 trainval1_tfr 27017
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train_tfr.sh 1 trainval1_tfr_test 27017
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train_tfr.sh 1 trainval1_tfr_bf 27017
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_vgg_hdf5_l2 27017
sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_vgg_hdf5_dot 27017 1 0
sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_vgg_hdf5_dot_slow 27017 1 1
