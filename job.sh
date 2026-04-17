#!/bin/bash
#SBATCH --job-name=train-model
#SBATCH --account=cs5814
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
# interact -A cs5814 -p a100_normal_q --gres=gpu:1 
python ./scripts/test.py