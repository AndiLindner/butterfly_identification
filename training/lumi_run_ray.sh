#!/bin/bash
set -euo pipefail


# At first time, have to install hyperopt in the container
#pip install hyperopt


# Basic tuning settings
#MODEL="convnext_large"
#MODEL_PATH="$RESULTS_DIR/convnext_large/10572172_2025-04-29_12-28/pytorch_model.bin"
MODEL="maxvit_t"
MODEL_PATH="$RESULTS_DIR/maxvit_t/10341785_2025-04-13_17-52/pytorch_model.bin"
#MODEL="regnet_x_32gf"
#MODEL_PATH="$RESULTS_DIR/regnet_x_32gf/10341927_2025-04-13_18-32/pytorch_model.bin"
BATCH_SIZE=128
EPOCHS=4
OVERSAMPLING=1  # 0 for False (using class weights) or 1 for True
NUM_SAMPLES=60  # Number of samples to draw from search config space


# Keep track of some settings
if [[ ${SLURM_PROCID} -eq 0 ]]; then
    echo
    echo "Model: $MODEL"
    echo "Model path: $MODEL_PATH"
    #echo "Number of GPUs: $(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_NODE))"
    echo "Batch size: $BATCH_SIZE"
    echo "Oversampling: $OVERSAMPLING"
    echo
fi


# Start the Ray cluster on all nodes
if [[ ${SLURM_PROCID} -eq 0 ]]; then
    ray start --head \
        --dashboard-host="0.0.0.0" --node-ip-address="$head_node_ip" \
        --port="$ray_gcs_port" --block &
elif [[ ${SLURM_LOCALID} -eq 0 ]]; then
        ray start --address="$RAY_ADDRESS" --block &
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
    --results-dir $RESULTS_DIR

