#!/bin/bash

##SBATCH --test-only

#SBATCH --job-name bfly

#SBATCH -D /home/anli100105/eurocc/projects/butterflies/andi
#SBATCH -o /home/anli100105/slurm_out/%A_%a_%x_%j.out
#SBATCH -e /home/anli100105/slurm_out/%A_%a_%x_%j.err

##SBATCH --mail-type=ALL,ARRAY_TASKS
##SBATCH --mail-user=XXXXXX

#SBATCH --account=p201004

#SBATCH --partition zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
##SBATCH --nodelist=n3005-027

#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=368  # max: 384-16 (2x8 for weka)
##SBATCH --cpus-per-task=184  # 192-8 (8 for weka) -> less mem for loading
#SBATCH --cpus-per-task=92  # even less mem for loading
#SBATCH --hint=nomultithread  # Avoid explicitly on MUSICA
##SBATCH --gpus-per-task=4  # not exported on MUSICA
#SBATCH --gpus-per-node=4
##SBATCH --cpus-per-gpu=92
##SBATCH --gres=gpu:4

#SBATCH --mem=682G  # 682G is max

#SBATCH --time=3-00:00:00  # 3 days is max

##SBATCH --array=0-40%10


# Using srun with multiple cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
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


module purge
module load EESSI/2023.06

# conda or pip
ENV=conda
echo
if [[ $ENV == "conda" ]]; then
    echo "Using the conda environment"
    module load Miniforge3
    eval "$(conda shell.bash hook)"
    conda activate bfly_env
else
    echo "Using the pip environment"
    source /weka/$SLURM_JOB_ACCOUNT/$USER/bfly_env/bin/activate
fi

#which python
#python --version
#python -c "import torch; print(torch.cuda.is_available())"
echo


# Decide to train or tune
TASK=train

if [[ $TASK == "tune" ]]; then
    echo "Tuning with Ray"
    worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
    head_node=${worker_list[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    #ray_gcs_port=$(($RANDOM%(35000-20000+1)+20000))
    ray_gcs_port=24897
    # Define the Ray address variable for all connections to the cluster
    export RAY_ADDRESS="$head_node_ip:$ray_gcs_port"
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" ray start --head \
        --node-ip-address="$head_node_ip" --port="$ray_gcs_port" --block &
    # Connect other nodes to the Ray cluster
    for i in $(seq 1 $(($SLURM_JOB_NUM_NODES-1)))
    do
        srun --overlap --nodes=1 --ntasks=1 -w "${worker_list[$i]}" ray start \
            --address="$RAY_ADDRESS" --block &
    done
    srun --overlap musica_run_ray.sh
else
    echo "Training with Accelerate"
    srun musica_run_acc.sh
fi


# Keep track of consumed resources
echo
sacct -j $SLURM_JOB_ID \
--format=jobname%15,jobid,state,elapsed,nnodes,ncpus,ntasks,cputime,maxrss,partition%20,nodelist

