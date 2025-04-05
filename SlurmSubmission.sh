#!/bin/bash
#SBATCH --nodes=1          # Use 1 Node     (Unless code is multi-node parallelized)
#SBATCH --ntasks=1
#SBATCH --account=survmeth622s101w25_class
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=3
#SBATCH -o slurm-%j.out-%N
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16000m
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gdesj@umich.edu   # Your email address has to be set accordingly
#SBATCH --job-name=SURVMETH_622_Assignment3        # the job's name you want to be used

module load python3.10-anaconda

export FILENAME=LLM_4_4.py
srun python FILENAME > SLURM_JOBID.out

echo "End of program at `date`"