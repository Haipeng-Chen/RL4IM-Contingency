#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-48:00          # Runtime in D-HH:MM, minimum of 10 minutes
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
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=400 p=0.025 method='lazy_adaptive_greedy' 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 mode='test' node_test=500 p=0.02 method='lazy_adaptive_greedy' # not finished yet

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 node_train=200 node_test=200 p=0.05 reward_type=0 #197
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 node_train=200 node_test=200 p=0.05 reward_type=1 #201
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 node_train=200 node_test=200 p=0.05 reward_type=2 #199

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 reward_type=1 #200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='train' node_train=300 node_test=300 p=0.033 reward_type=1 #198
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='train' node_train=400 node_test=400 p=0.025 reward_type=1 #202
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 mode='train' node_train=500 node_test=500 p=0.02 reward_type=1 #203

# Feb 9 5 pm 204 - 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.2 node_train=200 node_test=200 p=0.05 reward_type=3 #204


# Feb 9 9:25pm 205 - 209 #---------------------------------this set of results can be directly used
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 mode='test' node_test=100 p=0.1 method='lazy_adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=200 p=0.05 method='lazy_adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=300 p=0.033 method='lazy_adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 q=0.6 mode='test' node_test=400 p=0.025 method='lazy_adaptive_greedy' 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 mode='test' node_test=500 p=0.02 method='lazy_adaptive_greedy' 

# 210 - 214 #--------------------------this set of results can be directly used
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 mode='test' node_test=100 p=0.1 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=200 p=0.05 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=300 p=0.033 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 q=0.6 mode='test' node_test=400 p=0.025 method='adaptive_greedy' 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 mode='test' node_test=500 p=0.02 method='adaptive_greedy' 

# Feb 12 2pm 226 - 228
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 model_scheme='type2'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 model_scheme='type1'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 model_scheme='normal'


# ----------------------- starting formal experiments ---------------------------------
# Feb 12 11:26pm 230- 236
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=5
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=20
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=100
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=200
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=500






# these are not run
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 final_epsilon=0.05 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 epislon_decay_steps=6000
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='S2V_QN_2'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='GCN_QN_1'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='LINE_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='W2V_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 bs=64






