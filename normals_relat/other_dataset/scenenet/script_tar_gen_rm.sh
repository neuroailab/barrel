source activate env_torch_2
data_path=/om/user/chengxuz/Data/one_world_dataset/split_scenenet
script_path=/om/user/chengxuz/barrel/barrel/normals_relat/other_dataset/scenenet/get_normal_using_tf.py
cd ${data_path}
for indx in $(seq ${1} 3 16)
do
    srun tar -xf train_${indx}.tar.gz
    srun --mem=30000 --gres=gpu -c 5 -t 3-0:0:0 python ${script_path} --path ${data_path}/train/${indx}/ --hdf5 ${data_path}/hdf5s/train_${indx}.hdf5 --seed ${indx}
    srun rm -r ${data_path}/train/${indx}
done
