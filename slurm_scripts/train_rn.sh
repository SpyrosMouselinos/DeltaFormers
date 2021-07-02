#!/usr/bin/env bash
#SBATCH --job-name=train_rn
#SBATCH --partition=common
#SBATCH --qos=4gpu7d
#SBATCH --gres=gpu:1
#SBATCH --time=720
#SBATCH --output=./train_rn.log

echo "Training Relation Network"

python train.py --name experiment_rn --config config_rn.yaml

echo "Training of Relation Network Finished"