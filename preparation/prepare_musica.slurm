#!/bin/bash

##SBATCH --test-only

#SBATCH -J prep  # Job name

#SBATCH -o /home/anli100105/slurm_out/%A_%a_%x_%j.out
#SBATCH -e /home/anli100105/slurm_out/%A_%a_%x_%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXXXXX

#SBATCH --partition zen4_0768
#SBATCH --qos zen4_0768


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  # need only one core

#SBATCH --time=0-04:00:00

#SBATCH --mem=10G


module purge
module load EESSI/2023.06

# conda or pip
ENV=conda

if [[ $ENV == "conda" ]]; then
    module load Miniforge3
    eval "$(conda shell.bash hook)"
    conda activate bfly_env
else
    source /weka/$SLURM_JOB_ACCOUNT/$USER/bfly_env/bin/activate
fi


# Data directory with the "Schmetterlinge" folder
DATA_DIR="/weka/$SLURM_JOB_ACCOUNT/$USER"
# Create a folder with the symlinks of the actually used classes
mkdir -p $DATA_DIR/Schmetterlinge_sym

# Create directory with only relevant classes
python3 $DATA_DIR/preparation/create_symlinks.py --data-dir $DATA_DIR

# Create indices of training and validation set, plus test set
python3 $DATA_DIR/preparation/train_val-test-split.py --data-dir $DATA_DIR

# Create indices of dataset with oversampled minority classes
python3 $DATA_DIR/preparation/oversample.py --data-dir $DATA_DIR

