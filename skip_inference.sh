#!/bin/bash
#SBATCH --job-name=skip_connect
#SBATCH --account=cs5814
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=sonalis3@vt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# interact -A cs5814 -p a100_normal_q --gres=gpu:1 
python /home/sonalis3/arc3-ws/arc3/DL_project/skip-train.py