#!/usr/bin/sh

#SBATCH -J PA
#SBATCH -o train_proto_deeplatt_ucl_lp1_cel_200.out
#SBATCH -e train_proto_deeplatt_ucl_lp1_cel_200.out
#SBATCH -w node1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu1


source /home/wenyan6/.zshrc
conda activate py3
python train.py
