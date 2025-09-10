#!/bin/bash

# Script to load models and weights to the local cache directory.
# On LEONARDO to be executed on a login node with access to the internet.


#module load profile/deeplrn
#module load cineca-ai/4.3.0

source $WORK/$USER/venvs/bfly_env/bin/activate


mkdir -p ~/.cache/torch/hub/checkpoints

python3 load_models.py
