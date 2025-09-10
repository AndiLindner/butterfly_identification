#!/bin/bash

#SBATCH --job-name=prep   # Job name
#SBATCH --account=project_465001276  # Project for billing

#SBATCH --output=/users/lindnera/slurm_out/%j_%x_%n_%t.out
#SBATCH --error=/users/lindnera/slurm_out/%j_%x_%n_%t.err

#SBATCH --export=NONE  # new env, NIL for empty env

#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB

#SBATCH --time=0-09:00:00  # A bit more than 7h should suffice


# Export data directory with "Schmetterlinge" into container
export DATA_DIR=/flash/$SLURM_JOB_ACCOUNT/$USER/datasets

# Create folder for symlinks of actually used classes
mkdir -p $DATA_DIR/Schmetterlinge_sym

# Some PyTorch container with the necessary packages
export SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

singularity exec \
    -B $DATA_DIR \
    $SIF bash -c '$WITH_CONDA ; \
    python3 $DATA_DIR/preparation/create_symlinks.py --data-dir $DATA_DIR ; \
    python3 $DATA_DIR/preparation/train_val-test-split.py --data-dir $DATA_DIR ; \
    python3 $DATA_DIR/preparation/oversample.py --data-dir $DATA_DIR'
