#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-24:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu     # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --gres=gpu:1

# python rl4im.py --cascade 'DIC' --eps_decay True --eps_wstart 0.3 --propagate_p 0.3 --q 0.5 --use_cuda 1 --batch_size 128 
# python rl4im.py --cascade 'DIC' --eps_decay True --eps_wstart 0.3 --propagate_p 0.1 --q 1 --use_cuda 1 --batch_size 128 
# python rl4im.py --cascade 'DIC' --eps_decay True --eps_wstart 0 --propagate_p 0.3 --q 0.5 --use_cuda 1 --batch_size 128 
# python rl4im.py --cascade 'DIC' --eps_decay True --eps_wstart 0.2 --propagate_p 0.3 --q 0.5 --use_cuda 1 --batch_size 128
# python rl4im.py --cascade 'DIC' --eps_decay True --eps_wstart 0.3 --propagate_p 0.3 --q 1 --use_cuda 1 --batch_size 128
# python rl4im.py --cascade 'DIC' --eps_decay True --eps_wstart 0.3 --propagate_p 0.3 --q 0.5 --use_cuda 1 --batch_size 128
# python rl4im.py --cascade 'DIC' --eps_decay False --eps_wstart 0.3 --propagate_p 0.3 --q 0.5 --use_cuda 1 --batch_size 128

# Nov-30 Mon 3pm 
#python rl4im.py --graph_index 3 --cascade 'DIC' --eps_decay 1 --eps_wstart 0.3 --propagate_p 0.1 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 200 #fails because greedy_sample_size=200

# Nov-30 Mon 8:40pm
#python rl4im.py --graph_index 3 --cascade 'DIC' --eps_decay 1 --eps_wstart 0.3 --propagate_p 0.1 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 100 ############works well after changing 200 to 100
#python rl4im.py --graph_index 3 --cascade 'DIC' --eps_decay 1 --eps_wstart 0.3 --propagate_p 0.1 --q 1 --use_cuda 1 --batch_size 256 --greedy_sample_size 100

# Nov 30 11:26pm
#python rl4im.py --graph_index 2 --cascade 'LT' --eps_decay 1 --eps_wstart 0.3 --propagate_p 0.1 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 100
#python rl4im.py --graph_index 2 --cascade 'LT' --eps_decay 1 --eps_wstart 0.3 --propagate_p 0.1 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 500
#python rl4im.py --graph_index 2 --cascade 'SC' --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 100

# Dec 04 1:44pm 
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 20
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 100
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 50

# Dec 04 3.22pm 
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 256 --greedy_sample_size 50 # increase prob of choosing maxdegree in warm_start
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 50 
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.1 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 50
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 50 # warnstartL 0.33; 0.33; 0.33

# Dec 04 5.26pm
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.1 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 256 --greedy_sample_size 50 #change wstart prob back to 0.25, 0.25, 0.5
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.05 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 50 
#python rl4im.py --graph_index 3 --cascade 'SC' --budget_ratio 0.05 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 50


# Dec 05 5.29pm
#python rl4im.py --graph_index 2 --cascade 'SC' --budget_ratio 0.05 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 20
#python rl4im.py --graph_index 3 --cascade 'SC' --budget_ratio 0.05 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 20

#Dec 05 10:33pm
#python rl4im.py --graph_index 2 --cascade 'DIC' --l 0.1 --propagate_p 0.2 --d 1 --q 1 --budget_ratio 0.05 --eps_decay 1 --eps_wstart 0.3 --use_cuda 1 --batch_size 128 --greedy_sample_size 20 --num_episodes 50

# Dec 05 11:16pm 
#python rl4im.py --graph_index 3 --cascade 'SC' --budget_ratio 0.05 --eps_decay 1 --eps_wstart 0.3 --q 1 --use_cuda 1 --batch_size 128 --greedy_sample_size 50

# Dec 06 5pm important rl setting
#python rl4im.py --graph_index 2 --cascade 'DIC' --T 1 --l 0.1 --propagate_p 0.2 --d 1 --q 1 --budget 20 --eps_decay 1 --eps_wstart 0.3 --use_cuda 1 --batch_size 128 --greedy_sample_size 50 #lazy_greedy -> greedy


# Dec 06 10.48pm
#python rl4im.py --graph_index 3 --cascade 'DIC' --T 1 --l 0.1 --propagate_p 0.1 --d 1 --q 1 --budget 20 --eps_decay 1 --eps_wstart 0.3 --use_cuda 1 --batch_size 128 --greedy_sample_size 25
#python rl4im.py --graph_index 4 --cascade 'DIC' --T 1 --l 0.1 --propagate_p 0.1 --d 1 --q 1 --budget 20 --eps_decay 1 --eps_wstart 0.3 --use_cuda 1 --batch_size 128 --greedy_sample_size 25
#python rl4im.py --graph_index 3 --cascade 'DIC' --T 1 --l 0.1 --propagate_p 0.1 --d 1 --q 1 --budget 20 --eps_decay 1 --eps_wstart 0 --use_cuda 1 --batch_size 128 --greedy_sample_size 25 --num_episodes 200
python rl4im.py --graph_index 4 --cascade 'DIC' --T 1 --l 0.1 --propagate_p 0.1 --d 1 --q 1 --budget 20 --eps_decay 1 --eps_wstart 0 --use_cuda 1 --batch_size 128 --greedy_sample_size 25 --num_episodes 200



















