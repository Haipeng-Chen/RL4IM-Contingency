#!/bin/bash

# ---------------train RL4IM -----------------------
#for T in 8 12 16 20 24 
#do
#    sbatch gpu_run.sh -a $T
#done

#for BUDGET in 2 
#do
#    declare -i BUDGET
#    sbatch gpu_run.sh -b $BUDGET
#done

# ---------------test random ----------------------
MODE='test'
METHOD='random'
for T in 8 12 16 20 24
do
    sbatch gpu_run.sh -a $T -e $MODE -k $METHOD
done
