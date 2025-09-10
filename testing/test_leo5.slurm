#!/bin/bash 

# predict_testdata

#SBATCH --job-name=predict_testdata 

#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXXXXX
#SBATCH --output=predict_test.out
#SBATCH --error=predict_test.err

#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --time=00:18:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=25G

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun leo5_run_test.sh
