#!/bin/bash
#SBATCH --job-name=icdar_2017_jupyter
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --constraint="a6000|a5000|geforce_rtx_3090|geforce_rtx_2080"

#set -euo pipefail

cd /home/nachum/suliman/icdar_2017_script_classification

# Conda setup for non-interactive Slurm shell
source /home/nachum/suliman/anaconda/etc/profile.d/conda.sh
conda activate icdar_2017_sc

# Use writable Jupyter dirs
export JUPYTER_CONFIG_DIR=/home/nachum/suliman/jupyter/config
export JUPYTER_DATA_DIR=/home/nachum/suliman/jupyter/data
export JUPYTER_RUNTIME_DIR=/home/nachum/suliman/jupyter/runtime_${SLURM_JOB_ID}

mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"

# Helpful debug info
echo "HOSTNAME=$(hostname)"
echo "HOME=$HOME"
echo "PWD=$(pwd)"
echo "CONDA_PREFIX=$CONDA_PREFIX"

jupyter lab --no-browser --ip=0.0.0.0 --port=8888