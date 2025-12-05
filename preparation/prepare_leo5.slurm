#!/bin/bash

#SBATCH -J prep

#SBATCH -o /home/c102/cxxxxxx/slurm_out/%j_%x.out
#SBATCH -e /home/c102/cxxxxxx/slurm_out/%j_%x.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxx.xxx@uibk.ac.at

#SBATCH --partition=test
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --mem=12GB


module purge
module load cuda/12.6.2-gcc-13.3.0-n5c5eu7
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
module load Anaconda3/2023.10/miniconda-base-2023.10

eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"

ENV_NAME="bfly_env"

conda activate $SCRATCH/conda_envs/$ENV_NAME

# Data directory with the "Schmetterlinge" folder
DATA_DIR="/scratch/llm/datasets/butterfly_images"
# Create a folder with the symlinks of the actually used classes
mkdir -p $DATA_DIR/Schmetterlinge_sym

# Create directory with only relevant classes
python3 $DATA_DIR/preparation/create_symlinks.py --data-dir $DATA_DIR

# Create indices of training and validation set, plus test set
python3 $DATA_DIR/preparation/train_val-test-split.py --data-dir $DATA_DIR

# Create indices of dataset with oversampled minority classes
python3 $DATA_DIR/preparation/oversample.py --data-dir $DATA_DIR

