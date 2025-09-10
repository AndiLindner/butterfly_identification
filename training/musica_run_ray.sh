#!/bin/bash
set -euo pipefail


DATA_DIR=/weka/${SLURM_JOB_ACCOUNT}/${USER}
RESULTS_DIR=${DATA_DIR}/results
mkdir -p $RESULTS_DIR


# Basic tuning settings
MODEL="maxvit_t"
MODEL_PATH="$RESULTS_DIR/results/maxvit_t/18310_2025-06-17_12-45/pytorch_model.bin"
BATCH_SIZE=128
EPOCHS=4
OVERSAMPLING=1  # 0 for False (using class weights) or 1 for True
NUM_SAMPLES=32  # Number of samples to draw from search config space


# Keep track of some settings
if [[ ${SLURM_PROCID} -eq 0 ]]; then
    echo
    echo "Model: $MODEL"
    echo "Model path: $MODEL_PATH"
    #echo "Number of GPUs: $(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_TASK))"
    echo "Batch size: $BATCH_SIZE"
    echo "Oversampling: $OVERSAMPLING"
    echo
fi


python3 \
    run_ray.py \
    --model $MODEL \
    --model-path $MODEL_PATH \
    --batch-size $BATCH_SIZE \
    --num-samples $NUM_SAMPLES \
    --epochs $EPOCHS \
    --oversampling $OVERSAMPLING \
    --data-dir $DATA_DIR \
    --results-dir $RESULTS_DIR \

