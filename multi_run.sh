#!/bin/bash

# ---------------train RL4IM -----------------------
# 417-421
#BUDGET=2 
#for T in 2 4 6 8 10 
#do
#    sbatch gpu_run.sh -a $T -b $BUDGET
#done

# ---------------test random ----------------------
# 422-426
#MODE='test'
#METHOD='random'
#BUDGET=2
#for T in 2 4 6 8 10
#do
#    sbatch gpu_run.sh -a $T -e $MODE -k $METHOD -b $BUDGET
#done


# Feb 24 10pm tain on larger graphs # 427-431
#NODE_TRAIN=500
#NODE_TEST=500
#for T in 8 12 16 20 24
#do
#    sbatch gpu_run.sh -a $T -f $NODE_TRAIN -g $NODE_TEST 
#done



## Feb 24 10.30pm test random on larger graphs # 432-436
MODE='test'
METHOD='random'
NODE_TRAIN=500
NODE_TEST=500
for T in 8 12 16 20 24
do
    sbatch cpu_run.sh -a $T -e $MODE -k $METHOD -f $NODE_TRAIN -g $NODE_TEST 
done


# rerun test random 437-441
MODE='test'
METHOD='random'
BUDGET=2
for T in 2 4 6 8 10
do
    sbatch cpu_run.sh -a $T -e $MODE -k $METHOD -b $BUDGET
done
