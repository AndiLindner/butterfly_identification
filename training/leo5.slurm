#!/bin/bash

##SBATCH --test-only  # Only check slurm script
#SBATCH -J bfly  # Job name

#SBATCH -D /home/c102/cXXXXXX/eurocc/projects/butterflies/andi
#SBATCH -o /home/c102/cXXXXXX/slurm_out/%A_%a_%x_%j.out
#SBATCH -e /home/c102/cXXXXXX/slurm_out/%A_%a_%x_%j.err

#SBATCH --mail-type=ALL  #,ARRAY_TASKS
#SBATCH --mail-user=XXXXXX

#SBATCH --partition=std

#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=32  # Calculate 8-16 per GPU
#SBATCH --gpus-per-task=2  # To use SLURM variable, OR with specific type below
##SBATCH --cpus-per-gpu=8  # minimum 12 *per GPU* are good
##SBATCH --gpus=a30:2  # GPU type and count

##SBATCH --mem=50G
#SBATCH --mem-per-gpu=14G  # at least 14G, more for many models
#SBATCH --time=00:20:00

##SBATCH --array 6

# Very important to use multiple CPUs with srun
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
#export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_TASK))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Use all threads if two GPUs used

date

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
echo "GPUs per node:" $SLURM_GPUS_PER_NODE
echo "CPUs per GPU:" $SLURM_CPUS_PER_GPU
echo "Memory per node:" $SLURM_MEM_PER_NODE
echo "Number of array tasks:" $SLURM_ARRAY_TASK_COUNT
echo "Array task ID:" $SLURM_ARRAY_TASK_ID
echo

echo "All SLURM variables:"
env | grep 'SLURM_'
echo


srun leo5_run_acc.sh


<<'comment'
# Keep track of consumed resources
sleep 5
echo
sacct -j $SLURM_JOB_ID \
--format=jobname%15,jobid,state,elapsed,nnodes,ncpus,ntasks,cputime,maxrss,partition%20,nodelist
comment

