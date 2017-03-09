#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train.sh 1
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train_tfr.sh 1 trainval1_tfr 27017
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train_tfr.sh 1 trainval1_tfr_test 27017
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_%j.out ./script_train_tfr.sh 1 trainval1_tfr_bf 27017

#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_vgg_hdf5_l2 27017 0 0
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_vgg_hdf5_dot 27017 1 0
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_vgg_hdf5_dot_slow 27017 1 1
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_test_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_tfr_file_test 27017
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_rms_2 27017 1 1
#sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 trainval1_rms_center_3 27017 1
sbatch -J normalnet -o /om/user/chengxuz/slurm_out_all/slurm_normalnet_vgg_%j.out ./script_train_vgg_hdf5.sh 1 rms_momopt 27017 1
