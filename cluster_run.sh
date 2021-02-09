#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-24:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu     # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH -o output/%j.out  # File to which STDOUT will be written, %j inserts jobid

#155-156 #these models are used as the base model; theoretically these models are enough for all tasks
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=1
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=1 method='adaptive_greedy'

#157-161
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.8
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.8 method='adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 method='adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.4
#10

#162-167
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 node_train=500 node_test=1000 epislon_decay_steps=20000
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 node_test=1000 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 node_test=1000 method='lazy_adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.4 method='adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.2
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.2 method='adaptive_greedy'

#168-175
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 node_test=500 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 node_test=500 method='adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 node_test=1000 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 node_test=1000 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 node_test=1000 method='lazy_adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 epislon_decay_steps=20000
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 method='lazy_adaptive_greedy'

#Feb 6 11.55pm ET 178-181: to test reward_type
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.2 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.2 reward_type=1
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.2 reward_type=2
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.2 mode='test' method='adaptive_greedy'

# 182- 190
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode=‘train’ node_train=500 node_test checkpoint_path=./temp_dir/colge/sacred/159/models

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' checkpoint_path=./temp_dir/colge/sacred/159/models
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' method='adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 node_test=500 mode='test' checkpoint_path=./temp_dir/colge/sacred/159/models
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 node_test=500 mode='test' method='adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=500 checkpoint_path=./temp_dir/colge/sacred/159/models
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=500 method='adaptive_greedy'

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=1000 checkpoint_path=./temp_dir/colge/sacred/159/models
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=1000 method='adaptive_greedy'

#Feb 7 11:15pm 192-196 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 mode='test' node_test=200 p=0.05 method='lazy_adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=200 p=0.05 method='lazy_adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=300 p=0.033 method='lazy_adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=400 p=0.025 method='lazy_adaptive_greedy' # not finished yet
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 mode='test' node_test=500 p=0.02 method='lazy_adaptive_greedy' # not finished yet

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 node_train=200 node_test=200 p=0.05 reward_type=0 #197
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 node_train=200 node_test=200 p=0.05 reward_type=1 #201
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 node_train=200 node_test=200 p=0.05 reward_type=2 #199

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 reward_type=1 #200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='train' node_train=300 node_test=300 p=0.033 reward_type=1 #198
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='train' node_train=400 node_test=400 p=0.025 reward_type=1 #202
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 mode='train' node_train=500 node_test=500 p=0.02 reward_type=1 #203 not finished yet









# these are not run
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 final_epsilon=0.05 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 epislon_decay_steps=6000
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='S2V_QN_2'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='GCN_QN_1'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='LINE_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='W2V_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 bs=64






