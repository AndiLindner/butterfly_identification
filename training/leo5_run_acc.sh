#!/bin/bash
set -euo pipefail


# Model
MODELS=(resnet50 resnet101 resnet152
        wide_resnet50_2 wide_resnet101_2
        resnext101_32x8d resnext50_32x4d resnext101_64x4d
        regnet_y_16gf regnet_y_32gf regnet_y_128gf regnet_y_3_2gf
        regnet_x_8gf regnet_x_16gf regnet_x_32gf regnet_x_3_2gf
        densenet121 densenet169 densenet201 densenet161
        vgg19 vgg19_bn vgg16_bn
        efficientnet_v2_s efficientnet_v2_m efficientnet_v2_l
        vit_b_16 vit_b_32 vit_l_32 vit_l_16 vit_h_14
        swin_v2_t swin_v2_s swin_v2_b
        maxvit_t
        convnext_tiny convnext_small convnext_base convnext_large
        mobilenet_v3_small mobilenet_v3_large
        timm/maxvit_tiny_tf_224.in1k
        timm/maxvit_small_tf_224.in1k
        timm/maxvit_base_tf_224.in1k
        timm/maxvit_large_tf_224.in1k
    )

#MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL="mobilenet_v3_small"

# Training parameters
SCRIPT="run_acc.py"
BATCH_SIZE=16  # Batch size *per device*
EPOCHS=50
BASE_LR=0.004  # Scales linearly with devices in code
WEIGHT_DECAY=1e-6
OPTIMIZER="SGD"  # SGD, ASGD, RMSprop, Adam, AdamW, Adadelta, Adagrad
SCHEDULER="ConstantLR"  # StepLR, ReduceLROnPlateau
OVERSAMPLING=1  # 0 for False (using class weights) or 1 for True
CHECKPOINTING=0  # 0 for False or 1 for True (checkpoint after each epoch)


ENV="$SCRATCH/conda_envs/bfly_env"
DATA_DIR="/scratch/llm/datasets/butterfly_images"
RESULTS_DIR=$SCRATCH


module purge
module load cuda/12.6.2-gcc-13.3.0-n5c5eu7 >/dev/null # To compile CUDA ops
module load Anaconda3/2023.10/miniconda-base-2023.10 >/dev/null
module list

eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
conda activate $ENV


export MASTER_PORT=24998
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)

#export ACCELERATE_DEBUG_MODE="1"

# Keep track of settings
if [[ $(hostname) == "$MASTER_ADDR"* ]]; then
    echo
    echo "Model: $MODEL"
    echo "Number of GPUs: $(($SLURM_NNODES*$SLURM_GPUS_PER_TASK))"
    echo "Batch size per GPU: $BATCH_SIZE"
    #echo "Oversampling: $OVERSAMPLING"
    echo "Base learning rate (scales by number of GPUs): $BASE_LR"
    echo "Optimizer: $OPTIMIZER"
    echo "Scheduler: $SCHEDULER"
fi

accelerate launch \
    --multi_gpu \
    --same_network \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --num_processes=$(($SLURM_NNODES*$SLURM_GPUS_PER_TASK)) \
    --num_cpu_threads_per_process=$(($SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_TASK)) \
    --rdzv_backend=static `# c10d does not work on LEO5 and LEONARDO`\
    --mixed_precision="no" \
    --dynamo_backend="no" \
    $SCRIPT \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --epochs $EPOCHS \
    --base-lr $BASE_LR \
    --weight-decay $WEIGHT_DECAY \
    --oversampling $OVERSAMPLING \
    --checkpointing $CHECKPOINTING \
    --data-dir $DATA_DIR \
    --results-dir $RESULTS_DIR \
    #--checkpoint /path
    #--model-path /path
