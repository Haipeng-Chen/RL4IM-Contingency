#!/bin/bash
#SBATCH -n 1               # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-0:20          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu     # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/baseline_random_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --gres=gpu:0

#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.5 --baseline 'maxdegree'
python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.5 --baseline 'random'
