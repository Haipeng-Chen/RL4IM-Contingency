#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-3:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu     # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1

#110-115
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 use_state_abs=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 method='adaptive_greedy'

#125-127
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 q=0.5
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 q=0.5 use_state_abs=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 q=0.5 method='adaptive_greedy'

#128-130
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 q=0.5 node_test=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 q=0.5 node_test=200 use_state_abs=False 
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 q=0.5 node_test=200 method='adaptive_greedy'





#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 final_epsilon=0.05 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 epislon_decay_steps=6000
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='S2V_QN_2'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='GCN_QN_1'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='LINE_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='W2V_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 bs=64






