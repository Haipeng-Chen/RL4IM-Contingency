#!/bin/bash
#SBATCH -n 1               # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-2:00          # Runtime in D-HH:MM, minimum of 10 minutes
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
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'random' --graph_index 3 --budget_ratio 0.06; 
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'maxdegree' --graph_index 3 --budget_ratio 0.06; 
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 10

# run Nov 30 Mon 1.24pm
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 100
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 200
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 500
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 1000
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 1
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 10

# run Nov 30 Mon 3.35pm
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 10
#python env.py --cascade 'DIC' --propagate_p 0.1 --q 1 --baseline 'ada_greedy' --graph_index 3 --budget_ratio 0.06 --greedy_sample_size 100

# Nov 30 Mon 3.45pm
#python env.py --graph_index 4 --cascade 'DIC' --propagate_p 0.1 --q 1 --budget_ratio 0.06 --baseline 'ada_greedy' --greedy_sample_size 100
#python env.py --graph_index 4 --cascade 'DIC' --propagate_p 0.1 --q 1 --budget_ratio 0.06 --baseline 'ada_greedy' --greedy_sample_size 200
#python env.py --graph_index 4 --cascade 'DIC' --propagate_p 0.1 --q 1 --budget_ratio 0.06 --baseline 'ada_greedy' --greedy_sample_size 300
#python env.py --graph_index 4 --cascade 'DIC' --propagate_p 0.1 --q 1 --budget_ratio 0.06 --baseline 'random'
#python env.py --graph_index 4 --cascade 'DIC' --propagate_p 0.1 --q 1 --budget_ratio 0.06 --baseline 'maxdegree'

# Dec 01 00.05am 
#python env.py --graph_index 2 --cascade 'LT' --l  0.08 --q 1 --budget_ratio 0.06 --baseline 'ada_greedy' --num_simul 500 #added num_simul in env.step

# Dec 01 00:38am 
#python env.py --graph_index 2 --cascade 'SC' --q 1 --budget_ratio 0.06 --baseline 'ada_greedy' --num_simul 500 

# Dec 03 4.05pm
#python env.py --graph_index 2 --cascade 'SC' --q 1 --budget_ratio 0.1 --baseline 'maxdegree' --num_simul 500
#python env.py --graph_index 2 --cascade 'SC' --q 1 --budget_ratio 0.1 --baseline 'maxdegree' --num_simul 200
#python env.py --graph_index 2 --cascade 'SC' --q 1 --budget_ratio 0.1 --baseline 'maxdegree' --num_simul 100

# Dec 04 1.54pm
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 20
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 50
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 100
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'random' --num_simul 1000; python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'maxdegree' --num_simul 1000

# Dec 04 5.43pm
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 20 # changed it to normal adaptive greedy, previously ada_greey is lazy_aga_greedy
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 10
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.1 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 5


# Dec 04 11:25pm
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 20 
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'lazy_ada_greedy' --num_simul 1000 --greedy_sample_size 20
#python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'random' --num_simul 1000; python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'maxdegree' --num_simul 1000
#python env.py --graph_index 3 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'ada_greedy' --num_simul 1000 --greedy_sample_size 20 
#python env.py --graph_index 3 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'lazy_ada_greedy' --num_simul 1000 --greedy_sample_size 50
python env.py --graph_index 3 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'random' --num_simul 1000; python env.py --graph_index 2 --cascade 'SC' --l 0.1 --propagate_p 0.3 --d 1 --q 1 --budget_ratio 0.05 --baseline 'maxdegree' --num_simul 1000










