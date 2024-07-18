#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_repo
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

#SBATCH --output="run_output/output.log"
#SBATCH --error="run_output/error.log"

python3 wholepose/demo.py
