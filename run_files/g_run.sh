#!/bin/bash

#SBATCH --account=cvcs2024
#SBATCH --job-name=generic_run
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

#SBATCH --output="run_output/g_output.log"
#SBATCH --error="run_output/g_error.log"

$1