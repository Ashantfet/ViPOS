#!/bin/bash
#SBATCH --job-name= randomforest
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1            
#SBATCH --time=30:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --mem=12G                  # Increase memory if needed
#SBATCH --output=/scratch/kalidas_1/project/tmp/%x-%N-%j.out
#SBATCH --error=/scratch/kalidas_1/project/tmp/%x-%N-%j.err

## Load environment
source venv/bin/activate
module load cuda/11.4
python3 transformer/main.py

    