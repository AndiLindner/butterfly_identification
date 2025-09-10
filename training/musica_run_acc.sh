#!/bin/bash
set -euo pipefail

DATA_DIR=/weka/${SLURM_JOB_ACCOUNT}/${USER}
RESULTS_DIR=${DATA_DIR}/results
mkdir -p $RESULTS_DIR


export MASTER_PORT=24998
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)


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
MODEL="timm/maxvit_large_tf_224.in1k"
#MODEL_PATH="${RESULTS_DIR}/max_vit_large/29455_2025-07-10_19-17/pytorch_model.bin"
#MODEL="maxvit_t"
#MODEL="convnext_large"
#MODEL_PATH="${RESULTS_DIR}/maxvit_lumi_10341785/pytorch_model.bin"
#MODEL="efficientnet_v2_l"
#MODEL_PATH="${RESULTS_DIR}/lumi_efficientnet_v2_l/pytorch_model.bin"

# Training parameters
BATCH_SIZE=4  # Batch size *per GPU*
EPOCHS=250
BASE_LR=0.0002  # Scales linearly with devices in code
WEIGHT_DECAY=0
OPTIMIZER="SGD"  # SGD, ASGD, RMSprop, Adam, AdamW, Adadelta, Adagrad
SCHEDULER="None"  # None, StepLR, ReduceLROnPlateau, CosineAnnealingLR
OVERSAMPLING=0  # 0 for False (using class weights) or 1 for True
CHECKPOINTING=0  # 0 for False or 1 for True (checkpointing after each epoch)


# Keep track of some settings
if [[ $(hostname) == "$MASTER_ADDR"* ]]; then
    echo
    echo "Model: $MODEL"
    echo "Number of GPUs: $((${SLURM_JOB_NUM_NODES}*${SLURM_GPUS_PER_TASK}))"
    echo "Batch size per GPU: $BATCH_SIZE"
    #echo "Oversampling: $OVERSAMPLING"
    echo "Base learning rate (scales by number of GPUs): $BASE_LR"
    echo "Optimizer: $OPTIMIZER"
    echo "Scheduler: $SCHEDULER"
    echo "Weight decay: $WEIGHT_DECAY"
fi


# DPP with HF Accelerate launched with accelerate launch
accelerate launch \
    --multi_gpu \
    --same_network \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --num_processes=$(($SLURM_JOB_NUM_NODES*$SLURM_GPUS_PER_TASK)) \
    --num_cpu_threads_per_process=$(($SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_TASK)) \
    --rdzv_backend=static `# c10d or static`\
    --mixed_precision="no" \
    --dynamo_backend="no" \
    run_acc.py \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --base-lr $BASE_LR \
    --weight-decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --oversampling $OVERSAMPLING \
    --checkpointing $CHECKPOINTING \
    --data-dir $DATA_DIR \
    --results-dir $RESULTS_DIR \
    #--model-path $MODEL_PATH \
    #--checkpoint $RESULTS_DIR/$MODEL/...

