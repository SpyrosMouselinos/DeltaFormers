#!/usr/bin/env bash
#SBATCH --job-name=train_rn_fp
#SBATCH --partition=common
#SBATCH --qos=4gpu7d
#SBATCH --gres=gpu:1
#SBATCH --nodelist=asusgpu3
#SBATCH --time=840
#SBATCH --output=./train_rn_fp.log


echo "Training Relation Network Image"

python train.py --name experiment_rn_fp --config config_rn_fp.yaml

echo "Training of Relation Network Image Finished"