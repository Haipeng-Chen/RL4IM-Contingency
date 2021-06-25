#!/bin/bash

#
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

#### RL training
# Q1: ablation study of reward shaping and state-abstraction
for REWARD_TYPE  in 0 1 3
do
    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN; sleep 20;
done
#
USE_STATE_ABS=False
sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN; sleep 20;

# Q2: number of train graphs
for GRAPH_NBR_TRAIN in 10 50 100 200 500
do
    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN; sleep 20;
done

# Q3: vary q
for Q in 0.2 0.4 0.6 0.8 
do
    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN; sleep 20;
done

# Q4: vary T = 8, 4, 2, 1 (note that T in the paper is number of time steps, in the code T is number of time steps*budget)
for BUDGET in 1 2 4 8
do
    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN; sleep 20;
done

# Q5: vary |V|
for NODE_TRAIN in 50 100 200 500 1000
do
    NODE_TEST=NODE_TRAIN
    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN; sleep 20;
done

#### RL test, an example when q=0.2
MODE='test'
CHECK_POINT_PATH=./results/rl4im/sacred/113/models
LOAD_STEP=1580
Q=0.2
sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;

#### CHANGE/random, an example of Question 3
MODE='test'
for METHOD in 'lazy_adaptive_greedy' 'random'
do
for Q in 0.2 0.4 0.6 0.8 
do
    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -w $REWARD_TYPE -x $USE_STATE_ABS -o $NUM_SIMUL_TRAIN; sleep 20;
done
done



