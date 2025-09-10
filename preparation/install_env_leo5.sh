#!/bin/bash
set -eo pipefail

module purge
module load cuda/12.2.1-gcc-13.2.0-m4ekvjj
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
module load Anaconda3/2023.10/miniconda-base-2023.10

echo
module list
echo

eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"

ENV_NAME="bfly_env"

mamba env create -f bfly_env.yaml \
        --yes \
        --prefix $SCRATCH/conda_envs/$ENV_NAME  # install in extra dir

conda activate $SCRATCH/conda_envs/$ENV_NAME

echo
echo "Mamba list:"
mamba list
