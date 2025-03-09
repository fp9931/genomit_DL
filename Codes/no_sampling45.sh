#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1                   # 1 tasks
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=no_sampling45.err       # standard error file
#SBATCH --output=no_sampling45.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python no_sampling_symp45.py