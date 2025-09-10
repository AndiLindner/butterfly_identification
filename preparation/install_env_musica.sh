#!/bin/bash
set -euo pipefail

module load EESSI/2023.06

# This worked without loading a CUDA module, perhaps bc of GPUs on login node

# conda or pip
ENV=pip

if [[ $ENV == "pip" ]]; then
    ENV_DIR="/weka/p201004/$USER/bfly_env"
    mkdir -p $ENV_DIR
    python3 -m venv $ENV_DIR --upgrade-deps
    source $ENV_DIR/bin/activate
    pip3 install -r bfly_env.txt --force --no-cache-dir
elif [[ $ENV == "conda" ]]; then
    module load Miniforge3
    eval "$(conda shell.bash hook)"
    conda env create --name bfly_env --file bfly_env.yaml --yes
    # add --prefix or alter the default location to a large filesystem
else
    print "Choose either pip or conda as installation method."
fi
