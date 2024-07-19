#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=test_repo
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00

#SBATCH --output="run_output/output_frames.log"
#SBATCH --error="run_output/error_frames.log"

python3 gen_frames.py
