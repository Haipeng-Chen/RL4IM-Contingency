#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-24:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu     # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --gres=gpu:1

python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-4
