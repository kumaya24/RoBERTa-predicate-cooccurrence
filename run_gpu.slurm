#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=gpu_job_output_%j.log

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hf_env

# Run your python script
# (Using the automation script as an example)
echo "Starting Python script..."
./all_derivations.sh
echo "Script finished."
