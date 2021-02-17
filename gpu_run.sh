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



# 60 - 64
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.2 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.4 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=0.8 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 save_every=2 q=1.0 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0

# 65 - 68
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 save_every=2 q=0.6 mode='train' node_train=200 node_test=200 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 reward_type=0































#something place holder

