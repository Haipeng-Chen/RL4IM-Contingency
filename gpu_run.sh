#!/bin/bash
#SBATCH -n 1              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p Your_partition # Partition to submit to
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
REAL_GRAPH_NAME=None
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

CHECK_POINT_PATH=""
LOAD_STEP=0

############################################################################
ARGUMENT_LIST=(
    "platform"
    "gpuid"
    "NODE_TRAIN"
    "NODE_TEST"
    "Q"
    "T"
    "BUDGET"
    "GRAPH_TYPE"
    "IS_REAL_GRAPH"
    "SAMPLE_NODES_RATIO"
    "REAL_GRAPH_NAME"
    "MODE"
    "M"
    "PROPAGATE_P"
    "P"
    "CASCADE"
    "REWARD_TYPE"
    "USE_STATE_ABS"
    "GRAPH_NBR_TRAIN"
    "LR"
    "METHOD"
    "EPSILON_DECAY_STEPS"
    "NUM_SIMUL_TRAIN"
    "CHECK_POINT_PATH"
    "LOAD_STEP"
    "SAVE_EVERY"
    "GREEDY_SAMPLE_SIZE"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@" 
)

# set default value
# there are three ways: normal, slurm, docker
#   normal: run tasks with python (on your laptop or on remote machines)
#   slurm: run tasks with sbatch
#   docker: run tasks with docker
platform="normal"
gpuid=3

eval set --$opts

while true; do
    case "$1" in
    --platform)
        shift
        platform=$1
        ;;
    --gpuid)
        shift
        gpuid=$1
        ;;
    --NODE_TRAIN)
        shift
        NODE_TRAIN=$1
        ;;
    --NODE_TEST)
        shift
        NODE_TEST=$1
        ;;
    --Q)
        shift
        Q=$1
        ;;
    --T)
        shift
        T=$1
        ;;
    --BUDGET)
        shift
        BUDGET=$1
        ;;
    --GRAPH_TYPE)
        shift
        GRAPH_TYPE=$1
        ;;
    --SAMPLE_NODES_RATIO)
        shift
        SAMPLE_NODES_RATIO=$1
        ;;
    --REAL_GRAPH_NAME)
        shift
        REAL_GRAPH_NAME=$1
        ;;
    --MODE)
        shift
        MODE=$1
        ;;
    --M)
        shift
        M=$1
        ;;
    --PROPAGATE_P)
        shift
        PROPAGATE_P=$1
        ;;
    --P)
        shift
        P=$1
        ;;
    --CASCADE)
        shift
        CASCADE=$1
        ;;
    --REWARD_TYPE)
        shift
        REWARD_TYPE=$1
        ;;
    --USE_STATE_ABS)
        shift
        USE_STATE_ABS=$1
        ;;
    --GRAPH_NBR_TRAIN)
        shift
        GRAPH_NBR_TRAIN=$1
        ;;
    --LR)
        shift
        LR=$1
        ;;
    --METHOD)
        shift
        METHOD=$1
        ;;
    --EPSILON_DECAY_STEPS)
        shift
        EPSILON_DECAY_STEPS=$1
        ;;
    --CHECK_POINT_PATH)
        shift
        CHECK_POINT_PATH=$1
        ;;
    --SAVE_EVERY)
        shift
        SAVE_EVERY=$1
        ;;
    --GREEDY_SAMPLE_SIZE)
        shift
        GREEDY_SAMPLE_SIZE=$1
        ;;
    --)
        shift
        break
        ;;
    # *)
    #     shift
    #     break
    #     ;;
    esac
    shift
done
############################################################################

RUN_COMMAND="--config=rl4im \
    --env-config=basic_env \
    --results-dir=results \
    with \
    T=$T \
    budget=$BUDGET \
    save_every=$SAVE_EVERY \
    q=$Q \
    mode=$MODE \
    node_train=$NODE_TRAIN \
    node_test=$NODE_TEST \
    m=$M \
    propagate_p=$PROPAGATE_P \
    p=$P \
    method=$METHOD \
    greedy_sample_size=$GREEDY_SAMPLE_SIZE \
    graph_type=$GRAPH_TYPE \
    is_real_graph=$IS_REAL_GRAPH \
    sample_nodes_ratio=$SAMPLE_NODES_RATIO \
    cascade=$CASCADE \
    reward_type=$REWARD_TYPE \
    use_state_abs=$USE_STATE_ABS \
    graph_nbr_train=$GRAPH_NBR_TRAIN \
    num_simul_train=$NUM_SIMUL_TRAIN \
    real_graph_name=$REAL_GRAPH_NAME \
    lr=$LR \
    epsilon_decay_steps=$EPSILON_DECAY_STEPS"


if [ -z "$CHECK_POINT_PATH" ]
then
    echo "CHECK_POINT_PATH is empty"
else
    RUN_COMMAND="${RUN_COMMAND} checkpoint_path=$CHECK_POINT_PATH load_step=$LOAD_STEP"
fi


if [[ ${platform} == "docker" ]]; then
    echo ${RUN_COMMAND}
    bash run_interactive.sh ${gpuid} python main.py ${RUN_COMMAND}
elif [[ ${platform} == "normal" ]]; then
    CUDA_VISIBLE_DEVICES=${gpuid} python main.py ${RUN_COMMAND}
else
    python main.py ${RUN_COMMAND}
fi
