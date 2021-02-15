#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-72:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:0
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
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=500

# Feb 13 4pm  # 240 - 244 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.2 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=500
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.4 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=500
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.6 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=500
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=0.8 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=500
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=11 q=1.0 mode='train' node_train=200 node_test=200 p=0.05 graph_nbr_train=500

# Feb 13 4.30pm # 245 - 249
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.2 mode='test' node_test=200 p=0.05 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.4 mode='test' node_test=200 p=0.05 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=200 p=0.05 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.8 mode='test' node_test=200 p=0.05 method='adaptive_greedy'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=1.0 mode='test' node_test=200 p=0.05 method='adaptive_greedy'

#
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 mode='test' node_test=50 p=0.15 graph_nbr_train=500 graph_node_var=10 method='adaptive_greedy'

#--------------------------really start formal expeirments --------------------------
# Feb 14 11am  270-275 ############# these are useful models
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=5 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=20
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=100
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.6 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200

# Feb 14 5pm  291 - 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/270/models load_step=1746 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/271/models load_step=2016 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/272/models load_step=1908 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/273/models load_step=1440 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/274/models load_step=1872 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/275/models load_step=1332 graph_nbr_test=50

# Feb 14 5.30pm 297 - # use normalized rewards 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/270/models load_step=1746 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/271/models load_step=2016 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/272/models load_step=1908 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/273/models load_step=1440 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/274/models load_step=1872 graph_nbr_test=50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/275/models load_step=1332 graph_nbr_test=50

# Feb 14 5.40pm 306 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10

# 308 - 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/270/models load_step=1746 graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/271/models load_step=2016 graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/272/models load_step=1908 graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/273/models load_step=1440 graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/274/models load_step=1872 graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/275/models load_step=1332 graph_nbr_test=10

# Feb 14 9pm 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.2 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.4 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.8 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=1.0 mode='train' node_train=100 node_test=100 p=0.1 model_scheme='type1' graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200

# Feb 14 11pm  Exp 2.2
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.2 mode='train' node_train=100 node_test=100 p=0.1 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.4 mode='train' node_train=100 node_test=100 p=0.1 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.4 mode='train' node_train=100 node_test=100 p=0.1 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=0.8 mode='train' node_train=100 node_test=100 p=0.1 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False reward_type=0
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 save_every=2 q=1.0 mode='train' node_train=100 node_test=100 p=0.1 graph_node_var=20 epsilon_decay_steps=1000 graph_nbr_train=200 use_state_abs=False reward_type=0

# Feb 15 2.30am Exp 3.3
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=1 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=2 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=6 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=12 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10

# Feb 15 2.50am Exp 4.3 335 - 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=2 q=0.6 mode='test' node_test=50 p=0.2 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=9 budget=3 q=0.6 mode='test' node_test=100 p=0.1 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=200 p=0.05 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 q=0.6 mode='test' node_test=300 p=0.033 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=5 q=0.6 mode='test' node_test=400 p=0.025 graph_node_var=20 method='adaptive_greedy' graph_nbr_test=10





# these are not run
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 final_epsilon=0.05 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 epislon_decay_steps=6000
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='S2V_QN_2'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='GCN_QN_1'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='LINE_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 model='W2V_QN'
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-3 max_episodes=1 T=6 budget=2 bs=64






