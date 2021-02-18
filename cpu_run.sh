#!/bin/bash
#SBATCH -n 2              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue  # Partition to submit to
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
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=50 graph_node_var=20 method='random' graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=100 graph_node_var=20 method='random' graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 method='random' graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=500 graph_node_var=20 method='random' graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=1000 graph_node_var=20 method='random' graph_nbr_test=10;








































############
