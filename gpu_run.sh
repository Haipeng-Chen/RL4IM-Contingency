#!/bin/bash
#SBATCH -n 1              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu,seas_gpu,seas_dgx1 # Partition to submit to
#SBATCH --mem=10000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH -o output/%j.out  # File to which STDOUT will be written, %j inserts jobid


T=8
BUDGET=4
Q=0.6
NODE_TRAIN=200
NODE_TEST=200

GRAPH_TYPE='erdos_renyi'
IS_REAL_GRAPH=False
SAMPLE_NODES_RATIO=1
MODE='train'
REAL_GRAPH_NAME='India'
M=5
PROPAGATE_P=0.1
P=0.1
CASCADE='IC'

LR=0.001
METHOD='rl'
EPSILON_DECAY_STEPS=1000
SAVE_EVERY=20
GREEDY_SAMPLE_SIZE=100
NUM_SIMUL_TRAIN=100

while getopts f:g:q:t:b:c:d:u:v:e:h:i:j:k:l:m:n:o:p:r:s:z: option
do
case "${option}"
in
#problem settings (evaluate)
f) NODE_TRAIN=${OPTARG};;
g) NODE_TEST=${OPTARG};;
q) Q=${OPTARG};;
t) T=${OPTARG};;
b) BUDGET=${OPTARG};;
#other problem settings
c) GRAPH_TYPE=${OPTARG};;
d) IS_REAL_GRAPH=${OPTARG};;
u) SAMPLE_NODES_RATIO=${OPTARG};;
v) REAL_GRAPH_NAME=${OPTARG};;
e) MODE=${OPTARG};;
h) M=${OPTARG};;
i) PROPAGATE_P=${OPTARG};;
j) P=${OPTARG};;
k) CASCADE=${OPTARG};;
#methods related
l) LR=${OPTARG};;
m) METHOD=${OPTARG};;
n) EPSILON_DECAY_STEPS=${OPTARG};;
o) NUM_SIMUL_TRAIN=${OPTARG};;
p) CHECK_POINT_PATH=${OPTARG};;
r) LOAD_STEP=${OPTARG};;
s) SAVE_EVERY=${OPTARG};;
z) GREEDY_SAMPLE_SIZE=${OPTARG};;
esac
done

#echo 'mode is:' $MODE
#echo 'method is:' $METHOD
#echo 'budget is:' $BUDGET

python main.py --config=rl4im --env-config=basic_env --results-dir=results with T=$T budget=$BUDGET save_every=$SAVE_EVERY q=$Q mode=$MODE node_train=$NODE_TRAIN node_test=$NODE_TEST m=$M propagate_p=$PROPAGATE_P p=$P method=$METHOD greedy_sample_size=$GREEDY_SAMPLE_SIZE graph_type=$GRAPH_TYPE is_real_graph=$IS_REAL_GRAPH sample_nodes_ratio=$SAMPLE_NODES_RATIO cascade=$CASCADE real_graph_name=$REAL_GRAPH_NAME lr=$LR epsilon_decay_steps=$EPSILON_DECAY_STEPS num_simul_train=$NUM_SIMUL_TRAIN checkpoint_path=$CHECK_POINT_PATH load_step=$LOAD_STEP
