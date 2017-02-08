#!/bin/bash
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=35000
#SBATCH -t 7-0:0:0
#SBATCH -c 5
# Original training script
source activate env_torch_2
source ~/.tunnel
#python train_normalnet.py --expId trainval_om --seed ${1}
python train_normalnet${4}.py --expId ${2} --seed ${1} --cacheDirPrefix /om/user/chengxuz --nport ${3}
