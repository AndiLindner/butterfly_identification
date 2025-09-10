#!/bin/bash

#SBATCH -J prep  # Job name

#SBATCH -o /leonardo/home/userexternal/alindner/slurm_out/%j_%x.out
#SBATCH -e /leonardo/home/userexternal/alindner/slurm_out/%j_%x.err

#SBATCH --account=euhpc_d12_020
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXXXXX

#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal  #boost_qos_dbg

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  # need only one core, mem will be adjusted

#SBATCH --time=0-09:00:00  # Run time limit


module load profile/deeplrn
module load cineca-ai/4.3.0

# Data directory with the "Schmetterlinge" folder
DATA_DIR="$WORK/$USER/datasets"
# Create a folder with the symlinks of the actually used classes
mkdir -p $DATA_DIR/Schmetterlinge_sym

# Create directory with only relevant classes
python3 $DATA_DIR/preparation/create_symlinks.py --data-dir $DATA_DIR

# Create indices of training and validation set, plus test set
python3 $DATA_DIR/preparation/train_val-test-split.py --data-dir $DATA_DIR

# Create indices of dataset with oversampled minority classes
python3 $DATA_DIR/preparation/oversample.py --data-dir $DATA_DIR


