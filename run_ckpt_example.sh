python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=1 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/65/models load_step=1936 graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=2 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/66/models load_step=1808 graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=4 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/67/models load_step=1360 graph_nbr_test=10;
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=8 budget=8 q=0.6 mode='test' node_test=200 graph_node_var=20 checkpoint_path=./temp_dir/colge/sacred/75/models load_step=1382 graph_nbr_test=10;