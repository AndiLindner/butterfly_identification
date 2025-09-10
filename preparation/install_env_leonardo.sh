#!/bin/bash
set -euo pipefail

module load python/3.11.6--gcc--8.5.0

ENV_DIR="$WORK/$USER/venvs/bfly_env"
mkdir -p $ENV_DIR
python3 -m venv $ENV_DIR --upgrade-deps
source $ENV_DIR/bin/activate

# Install packages
pip3 install -r bfly_env.txt --force --no-cache-dir
