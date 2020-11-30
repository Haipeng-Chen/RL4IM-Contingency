#!/bin/bash
#SBATCH -n 1               # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-0:40          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu     # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/baseline_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --gres=gpu:0

# run Nov 24 Tues 11:56am 
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.6 --baseline 'random';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.6 --baseline 'maxdegree';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.6 --baseline 'ada_greedy';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.7 --baseline 'random';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.7 --baseline 'maxdegree';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.7 --baseline 'ada_greedy';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.8 --baseline 'random';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.8 --baseline 'maxdegree';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.8 --baseline 'ada_greedy';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.9 --baseline 'random';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.9 --baseline 'maxdegree';
#python env.py --cascade 'DIC' --propagate_p 0.3 --q 0.9 --baseline 'ada_greedy';

# run Nov 30 Mon 12:38pm
python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'random' --graph_index 3 --budget_ratio 0.06; 
python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'maxdegree' --graph_index 3 --budget_ratio 0.06; 
python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 10
