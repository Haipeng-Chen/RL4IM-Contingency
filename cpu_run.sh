#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output/%j.out  # File to which STDOUT will be written, %j inserts jobid

# 
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.2 mode='test' node_test=200 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.4 mode='test' node_test=200 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.6 mode='test' node_test=200 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.8 mode='test' node_test=200 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=1.0 mode='test' node_test=200 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10

# 23 - 27
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.2 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.4 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.8 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=1.0 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10

#28 - 32
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.2 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.4 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=0.8 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=6 budget=3 q=1.0 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;

# 43 - 46
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10

# 47 48 49 50
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10

# 56 57 58 59  Exp 3.
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10

# 68 - 72 Exp 4. change
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=50 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=100 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=500 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=20 method='lazy_adaptive_greedy' graph_nbr_test=10

#  Exp 4 celf 93 94 95 96 97

#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=50 graph_node_var=20 method='greedy' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=100 graph_node_var=20 method='greedy' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='greedy' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=500 graph_node_var=20 method='greedy' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=20 method='greedy' graph_nbr_test=10;

# 73 -78, one of them is baseRL from Q3
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=50 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=100 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=500 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=20 method='random' graph_nbr_test=10;

# 139 - 143
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=50 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=100 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=500 graph_node_var=20 method='random' graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=20 method='random' graph_nbr_test=10;

# random  204 205
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode=test graph_node_var=0 method='random' graph_nbr_test=10 sample_graph=True sample_graph_name=India;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode=test graph_node_var=0 method='random' graph_nbr_test=10 sample_graph=True sample_graph_name=Exhibition;

# change 206 207
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode=test graph_node_var=0 method=lazy_adaptive_greedy graph_nbr_test=10 sample_graph=True sample_graph_name=India;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode=test graph_node_var=0 method=lazy_adaptive_greedy graph_nbr_test=10 sample_graph=True sample_graph_name=Exhibition

# High-School 216 217
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode=test graph_node_var=0 method= lazy_adaptive_greedy graph_nbr_test=10 sample_graph=True sample_graph_name=High-School;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=2 q=0.6 mode=test graph_node_var=0 method='random' graph_nbr_test=10 sample_graph=True sample_graph_name=High-School


# run time of CHANGE
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False


# random when increase T
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=8 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=12 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=16 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=20 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=24 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=28 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=32 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False


#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=0 method='lazy_adaptive_greedy' graph_nbr_test=1 sample_graph=False


#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=28 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=32 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False


#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=4 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=28 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=32 budget=4 save_every=5 q=0.6 m=4 mode=test node_test=1000 graph_node_var=0 method=random sample_graph=False propagate_p=0.15


#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=32 budget=4 save_every=5 q=0.6 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False

# Feb 19 1pm random on dense graphs 280 281 282 283 284
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 m=5 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 m=5 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 q=0.6 m=5 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=4 q=0.6 m=5 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 q=0.6 m=5 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10



# Feb 19 5pm change m  293 - 297
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10;
#python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.2 graph_nbr_test=10

# Feb 19 8pm change propagate_p to 0.15 #306 - 310
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.15 graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=12 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.15 graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=16 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.15 graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=20 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.15 graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=24 budget=4 q=0.6 m=4 mode=test node_test=200 graph_node_var=20 method=random sample_graph=False propagate_p=0.15 graph_nbr_test=10






































############
