#!/bin/bash
set -euo pipefail


# LEONARDO specifics
DATA_DIR="$WORK/$USER/datasets"
RESULTS_DIR="$WORK/$USER/butterflies/results"
mkdir -p $RESULTS_DIR

# Basic tuning settings
MODEL="convnext_large"
MODEL_PATH="$RESULTS_DIR/convnext_large_lumi_10572172/pytorch_model.bin"
#MODEL="maxvit_t"
#MODEL_PATH="$RESULTS_DIR/maxvit_lumi_10341785/pytorch_model.bin"
#MODEL="regnet_x_32gf"
#MODEL_PATH="$RESULTS_DIR/regnet_x_32gf_lumi_10341927/pytorch_model.bin"
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

