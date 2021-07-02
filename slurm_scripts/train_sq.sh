#!/usr/bin/env bash
#SBATCH --job-name=train_sq
#SBATCH --partition=common
#SBATCH --qos=4gpu7d
#SBATCH --gres=gpu:1
#SBATCH --time=720
#SBATCH --output=./train_sq.log

echo "Training SQ Network"

python train.py --name experiment_sq --config config_sq.yaml

echo "Training of SQ Network Finished"