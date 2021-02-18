#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu  # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH -o output/%j.out  # File to which STDOUT will be written, %j inserts jobid



# Feb 16 1pm 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.005
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.4 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.005
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.005
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.005
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.005

# Feb 16 1pm 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.4 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001


#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1500 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=5000 max_global_t=8000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=5000 max_global_t=8000 graph_nbr_train=200 lr=0.0001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=5000 max_global_t=8000 graph_nbr_train=200 lr=0.0005
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 max_global_t=4000 graph_nbr_train=200 lr=0.001


#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.002;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.4 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.002;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.002;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.002;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.002

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 graph_nbr_train=200 lr=0.002 max_global_t=3000;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.4 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 graph_nbr_train=200 lr=0.002 max_global_t=3000;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 graph_nbr_train=200 lr=0.002 max_global_t=3000;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 graph_nbr_train=200 lr=0.002 max_global_t=3000;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 graph_nbr_train=200 lr=0.002 max_global_t=3000


#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.4 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001

# 34 35
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.005
# 36 37 39 40
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.005
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 graph_nbr_train=200 lr=0.001 max_global_t=3000 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.01

#  41
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 graph_nbr_train=200 lr=0.001 max_global_t=3000

# 52 53 54 55
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.0001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.0005

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.0001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.0005



# 60 - 64 Exp 2.1
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.4 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0

# 65 66 67  Exp 3.2 missed 1?
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0

# Exp 4
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir/Feb17baserl with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=50 node_test=50 graph_node_var=20 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir/Feb17baserl with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 graph_node_var=20 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir/Feb17baserl with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir/Feb17baserl with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=500 node_test=500 graph_node_var=20 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir/Feb17baserl with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=1000 node_test=1000 graph_node_var=20 graph_nbr_train=200 reward_type=0

# Exp 3.2 rerun
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0


# Exp 2.2.1 test baseRL 79 80 81 82 83
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.2 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/60/models load_step=1908 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.4 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/62/models load_step=1776 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/61/models load_step=1212 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.8 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/63/models load_step=1920 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=1.0 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/64/models load_step=1992 graph_nbr_test=10;

# Exp 3.2.1 test baseRL 84 85 86 88
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/65/models load_step=1936 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/66/models load_step=1808 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/67/models load_step=1360 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/75/models load_step=1382 graph_nbr_test=10;

# Exp 0 ablation study 89 90 91 92
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=1

# 115 - 116
# retrain RL4IM 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 lr=0.001
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=2000 max_global_t=4000 graph_nbr_train=200 lr=0.001

# retrain ablation study 117 - 120 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=1

# retrain ablation q=0.2  121 - 124 may mix up with above
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=1


# test RL4IM  Exp 2.1.1 125 126 127 128
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.2 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/12/models load_step=1560 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.4 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/13/models load_step=1920 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/37/models load_step=1632 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.8 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/115/models load_step=1332 graph_nbr_test=10;


# train RL4IM Exp 3.1 130 131 132 133 (129 also seems to be budget 1)
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200



# train RL4IM 4.1 134 135 136 137 138 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=50 node_test=50 graph_node_var=20 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 graph_node_var=20 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=500 node_test=500 graph_node_var=20 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=1000 node_test=1000 graph_node_var=20 graph_nbr_train=200


# test RL4IM 4.1.1 144 - 148
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=50 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/134/models load_step=1248 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=100 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/135/models load_step=1904 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/136/models load_step=1168 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=500 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/137/models load_step=1824 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/138/models load_step=1152 graph_nbr_test=10;

# test baseRL 4.2.1 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=50 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/110/models load_step=1680 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=100 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/111/models load_step=1184 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/112/models load_step=1664 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=500 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/113/models load_step=1856 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/114/models load_step=1888 graph_nbr_test=10;


# case study  6 
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=2 save_every=2 q=0.6 mode='train' node_train=20 node_test=20 graph_node_var=2 graph_nbr_train=200




































#something place holder
