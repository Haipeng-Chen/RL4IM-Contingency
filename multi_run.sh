#!/bin/bash

### THE FOLLOWING IS FOR SELECTING WHICH WAY TO RUN YOUR TASKS
# Read command line options
ARGUMENT_LIST=(
    "platform"
    "gpuid"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

#echo $opts

# set default value
# there are three ways: normal, slurm, docker
#   normal: run tasks with python (on your laptop or on remote machines)
#   slurm: run tasks with ${runner}
#   docker: run tasks with docker
platform="normal"
gpuid=0

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
      --)
        shift
        break
        ;;
    esac
    shift
done

runner=${runner}
if [[ ${platform} == "normal" ]]; then
    runner=bash
elif [[ ${platform} == "slurm" ]]; then
    runner=${runner}
elif [[ ${platform} == "docker" ]]; then
    runner=bash
else
    echo "Please select the right running platform: normal, slurm or docker"
fi


### THE FOLLOWING ARE HYPERPARAMETERS
NODE_TRAIN=200
NODE_TEST=200
Q=0.6
T=8
BUDGET=4
#
GRAPH_TYPE='powerlaw'
IS_REAL_GRAPH=False
MODE='train'
M=3
PROPAGATE_P=0.1
P=0.05
CASCADE='IC'
#
REWARD_TYPE=3
USE_STATE_ABS=True
GRAPH_NBR_TRAIN=200
#
NUM_SIMUL_TRAIN=200
EPSILON_DECAY_STEPS=1500
LR=0.003

TASK_PLATFORM=${platform}
GPU_ID=${gpuid}

BASE_CMD="--Q $Q \
          --T $T \
          --MODE $MODE \
          --NODE_TRAIN $NODE_TRAIN \
          --NODE_TEST $NODE_TEST \
          --P $P \
          --M $M \
          --PROPAGATE_P $PROPAGATE_P \
          --BUDGET $BUDGET \
          --GRAPH_TYPE $GRAPH_TYPE \
          --IS_REAL_GRAPH $IS_REAL_GRAPH \
          --SAMPLE_NODES_RATIO $SAMPLE_NODES_RATIO \
          --REAL_GRAPH_NAME $REAL_GRAPH_NAME \
          --CASCADE $CASCADE \
          --EPSILON_DECAY_STEPS $EPSILON_DECAY_STEPS \
          --LR $LR \
          --REWARD_TYPE $REWARD_TYPE \
          --NUM_SIMUL_TRAIN $NUM_SIMUL_TRAIN \
          --platform ${TASK_PLATFORM} \
          --gpuid ${GPU_ID}"

#### RL training
Q1: ablation study of reward shaping and state-abstraction
RL_TRAIN_CMD_ABL_ALL="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS"
for REWARD_TYPE  in 0 1 3
do  
    ${runner} gpu_run.sh ${RL_TRAIN_CMD_ABL_ALL} --REWARD_TYPE ${REWARD_TYPE}; sleep 20;
done
#
USE_STATE_ABS=False
RL_TRAIN_CMD_ABL_REST="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS"
${runner} gpu_run.sh ${RL_TRAIN_CMD_ABL_REST}; sleep 20;

# change abck to True
USE_STATE_ABS=True
# Q2: number of train graphs
RL_TRAIN_CMD_Q2="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS"
for GRAPH_NBR_TRAIN in 10 50 100 200 500
do
    ${runner} gpu_run.sh ${RL_TRAIN_CMD_Q2} --GRAPH_NBR_TRAIN ${GRAPH_NBR_TRAIN}; sleep 20;
done

# Q3: vary q
RL_TRAIN_CMD_Q3="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS"
for Q in 0.2 0.4 0.6 0.8 
do
    ${runner} gpu_run.sh ${RL_TRAIN_CMD_Q3} --Q ${Q}; sleep 3;
done

# Q4: vary T = 8, 4, 2, 1 (note that T in the paper is number of time steps, in the code T is number of time steps*budget)
RL_TRAIN_CMD_Q4="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS"
for BUDGET in 1 2 4 8
do
    ${runner} gpu_run.sh ${RL_TRAIN_CMD_Q4} --BUDGET ${BUDGET}; sleep 20;
done

# Q5: vary |V|
RL_TRAIN_CMD_Q5="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS"
for NODE_TRAIN in 50 100 200 500 1000
do
    NODE_TEST=NODE_TRAIN
    ${runner} gpu_run.sh ${RL_TRAIN_CMD_Q5} --NODE_TEST ${NODE_TEST} --NODE_TRAIN ${NODE_TRAIN}; sleep 20;
done

# #### RL test, an example when q=0.2
MODE='test'
CHECK_POINT_PATH=./results/rl4im/sacred/113/models
LOAD_STEP=1580
Q=0.2
RL_TRAIN_CMD_TEST="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS --CHECK_POINT_PATH $CHECK_POINT_PATH --LOAD_STEP $LOAD_STEP"
${runner} gpu_run.sh ${RL_TRAIN_CMD_TEST} --MODE ${MODE}; sleep 20;

# #### CHANGE/random, an example of Question 3
MODE='test'
RL_TRAIN_CMD_TEST2="${BASE_CMD} --USE_STATE_ABS $USE_STATE_ABS"
for METHOD in 'lazy_adaptive_greedy' 'random'
do
for Q in 0.2 0.4 0.6 0.8 
do
    ${runner} gpu_run.sh ${RL_TRAIN_CMD_TEST2} --MODE ${MODE} --METHOD ${METHOD} --Q ${Q}; sleep 20;
done
done
