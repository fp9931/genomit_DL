#!/bin/bash
#SBATCH --nodes=2                    # 1 node
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=output.err       # standard error file
#SBATCH --output=output.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python classification_DL_Pytorch.py