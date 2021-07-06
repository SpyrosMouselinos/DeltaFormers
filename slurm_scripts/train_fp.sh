#!/usr/bin/env bash
#SBATCH --job-name=train_fp
#SBATCH --partition=common
#SBATCH --qos=4gpu7d
#SBATCH --gres=gpu:1
#SBATCH --nodelist=asusgpu1
#SBATCH --time=840
#SBATCH --output=./train_fp.log


echo "Training Relation Network Image"

python train.py --name experiment_fp --config config_fp.yaml

echo "Training of Relation Network Image Finished"