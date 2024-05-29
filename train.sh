#!/bin/bash


#SBATCH --mail-user=jkyeaton@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=baseline_amfam
#SBATCH --output=test_runs/%j.%N.stdout
#SBATCH --error=test_runs/%j.%N.stderr
#SBATCH --chdir=/home/jkyeaton/2023-autumn-amfam/baseline/
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate unicom_env

export CONFIG_PATH="baseline_config.json"
export CHECKPOINT="/net/projects/amfam/baseline/second25_transfer_is224.pth"
export OUT_DIR="test_runs"

# Pass CONFIG_PATH and OUT_DIR to the python script
python3 baseline_reproduce.py --config ${CONFIG_PATH} --out_dir ${OUT_DIR} --transfer --checkpoint ${CHECKPOINT}