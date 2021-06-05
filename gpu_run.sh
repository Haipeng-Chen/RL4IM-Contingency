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

REWARD_TYPE=3
USE_STATE_ABS=True
GRAPH_NBR_TRAIN=200

NUM_SIMUL_TRAIN=200
LR=0.001
METHOD='rl'
EPSILON_DECAY_STEPS=1000
SAVE_EVERY=20
GREEDY_SAMPLE_SIZE=100

while getopts f:g:q:t:b:c:d:u:v:e:h:i:j:k:w:x:y:o:l:m:n:p:r:s:z: option
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
#methods (evaluate)
w) REWARD_TYPE=${OPTARG};;
x) USE_STATE_ABS=${OPTARG};;
y) GRAPH_NBR_TRAIN=${OPTARG};;
#methods (others)
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

python main.py --config=rl4im --env-config=basic_env --results-dir=results with T=$T budget=$BUDGET save_every=$SAVE_EVERY q=$Q mode=$MODE node_train=$NODE_TRAIN node_test=$NODE_TEST m=$M propagate_p=$PROPAGATE_P p=$P method=$METHOD greedy_sample_size=$GREEDY_SAMPLE_SIZE graph_type=$GRAPH_TYPE is_real_graph=$IS_REAL_GRAPH sample_nodes_ratio=$SAMPLE_NODES_RATIO cascade=$CASCADE reward_type=$REWARD_TYPE use_state_abs=$USE_STATE_ABS graph_nbr_train=$GRAPH_NBR_TRAIN num_simul_train=$NUM_SIMUL_TRAIN real_graph_name=$REAL_GRAPH_NAME lr=$LR epsilon_decay_steps=$EPSILON_DECAY_STEPS checkpoint_path=$CHECK_POINT_PATH load_step=$LOAD_STEP

