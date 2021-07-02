#!/usr/bin/env bash
#SBATCH --job-name=train_q
#SBATCH --partition=common
#SBATCH --qos=4gpu7d
#SBATCH --gres=gpu:1
#SBATCH --time=60
#SBATCH --output=./train_q.log

echo "Training Q Network"

python train.py --name experiment_q --config config_q.yaml

echo "Training of Q Network Finished"