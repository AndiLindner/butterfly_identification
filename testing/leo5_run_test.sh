#!/bin/bash

ENV="/scratch/cXXXXXX/conda/envs/pytorch_cuda"
DATA_DIR="/scratch/cXXXXXX/"
MODEL="maxvit_t" #"maxvit_t" or "convnext_large"
MODEL_DIR="/scratch/cXXXXXX/model_comp/final_models/MaxViT_tiny/finetuning/pytorch_model.bin"
RESULTS_DIR="/scratch/cXXXXXX/model_comp/model_test/results_maxvit_t"

BATCH_SIZE=64
NUM_WORKERS=$SLURM_CPUS_PER_TASK  # or e.g. OMP_NUM_THREADS

module purge
module load cuda/12.2.1-gcc-13.2.0-m4ekvjj  # To compile CUDA ops
module load Anaconda3/2023.10/miniconda-base-2023.10
eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate $ENV

python model_test.py\
	--data-dir $DATA_DIR \
	--model $MODEL \
	--model-dir $MODEL_DIR \
	--results-dir $RESULTS_DIR \
	--batch-size $BATCH_SIZE \
	--num-workers $NUM_WORKERS \
