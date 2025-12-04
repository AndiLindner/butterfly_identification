#!/bin/bash

##SBATCH --test-only  # Only check slurm script
#SBATCH -J bfly  # Job name

#SBATCH -o /leonardo/home/userexternal/alindner/slurm_out/%A_%a_%x_%j.out
#SBATCH -e /leonardo/home/userexternal/alindner/slurm_out/%A_%a_%x_%j.err

#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=XXXXXX

#SBATCH --account=euhpc_d14_048

#SBATCH --partition=boost_usr_prod

##SBATCH --qos=normal
##SBATCH --qos=boost_qos_dbg  # only 30 min runtime
#SBATCH --qos=boost_qos_lprod  # max 4 days runtime

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # only 1 per node task for torchrun
#SBATCH --cpus-per-task=32  # max 32 per node
##SBATCH --gres=gpu:4  # max 4 GPUs per node
#SBATCH --gpus-per-node=4  # GPUs per /node
##SBATCH --gpus-per-task=4  # not exported anymore
##SBATCH --cpus-per-gpu=8  # 32/4=8 cores per GPU

##SBATCH --mem=30GB  # Memory per node in K/M/G/T out of 481GB
##SBATCH --mem-per-gpu=16G

##SBATCH --time=0-00:30:00
##SBATCH --time=0-08:00:00
#SBATCH --time=4-00:00:00

#SBATCH --array=41-44
##SBATCH -d afterok:13135318  # start only after prep job done successfully


# Using srun with multiple cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
#export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_TASK))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # For data loader

# Explicitly export the number of GPUs per task
export SLURM_GPUS_PER_TASK=$SLURM_GPUS_PER_NODE

date
echo


# Keep track of acquired resources
echo
echo "Account:" $SLURM_JOB_ACCOUNT
echo "Job ID:" $SLURM_JOB_ID ", Job name:" $SLURM_JOB_NAME
echo "Cluster:" $SLURM_CLUSTER_NAME
echo "Partition:" $SLURM_JOB_PARTITION
echo "QOS:" $SLURM_JOB_QOS
echo "Number of nodes:" $SLURM_JOB_NUM_NODES
echo "Nodes:" $SLURM_JOB_NODELIST
echo "Node:" $SLURMD_NODENAME
echo "Number of tasks:" $SLURM_NTASKS
echo "Tasks per node:" $SLURM_TASKS_PER_NODE
echo "CPUs per task:" $SLURM_CPUS_PER_TASK
echo "GPUs per task:" $SLURM_GPUS_PER_TASK
echo "CPUs per GPU:" $SLURM_CPUS_PER_GPU
echo "Memory per node:" $SLURM_MEM_PER_NODE
echo "Number of array tasks:" $SLURM_ARRAY_TASK_COUNT
echo "Array task ID:" $SLURM_ARRAY_TASK_ID
echo

echo "All SLURM variables:"
env | grep 'SLURM_'
echo


# Activate Python environment
source $WORK/$USER/venvs/bfly_env/bin/activate


# Decide to train or tune
TASK=train

if [[ $TASK == "tune" ]]; then
    echo "Tuning with Ray"
    module load python/3.11.6--gcc--8.5.0 >/dev/null
    module load cuda/12.1 >/dev/null
    source $WORK/$USER/venvs/bfly_env/bin/activate
    worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
    head_node=${worker_list[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    ray_gcs_port=$(($RANDOM%(35000-20000+1)+20000))
    # Define the Ray address variable for all connections to the cluster
    export RAY_ADDRESS="$head_node_ip:$ray_gcs_port"
    # Note that ntasks in srun is set to 1 because otherwise the ray start
    # command would be executed ntasks times, spawning ntasks*--num-cpus
    # workers.
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" ray start --head \
        --dashboard-host="0.0.0.0" --node-ip-address="$head_node_ip" \
        --port="$ray_gcs_port" --block &
    # Connect other nodes to the Ray cluster
    for i in $(seq 1 $(($SLURM_JOB_NUM_NODES-1)))
    do
        srun --overlap --nodes=1 --ntasks=1 -w "${worker_list[$i]}" ray start \
            --address="$RAY_ADDRESS" --block &
    done
    #ray_dashboard_port=$(($RANDOM%(50000-35000+1)+35000))
    #echo [INFO]: Ray cluster is running.
    srun --overlap leonardo_run_ray.sh
else
    echo "Training with Accelerate"
    # Load Cineca-AI module with most required software
    #module load profile/deeplrn > /dev/null
    #module load cineca-ai/4.3.0 > /dev/null
    srun leonardo_run_acc.sh
fi

