#!/bin/bash

# ---------------train RL4IM -----------------------
# 417-421  !!!!!!!!!!!!!!!!!!!!!
#BUDGET=2 
#for T in 2 4 6 8 10 
#do
#    sbatch gpu_run.sh -t $T -b $BUDGET
#done

# ---------------test random ----------------------
# 422-426 #incorrect run 
#MODE='test'
#METHOD='random'
#BUDGET=2
#for T in 2 4 6 8 10
#do
#    sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -b $BUDGET
#done


# Feb 24 10pm tain on larger graphs # 427-431 !!!!!!!!!!!!!!!
#NODE_TRAIN=500
#NODE_TEST=500
#for T in 8 12 16 20 24
#do
#    sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST 
#done



## Feb 25 4.30pm test random on larger graphs # 465-469
#MODE='test'
#METHOD='random'
#NODE_TRAIN=500
#NODE_TEST=500
#for T in 8 12 16 20 24
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST 
#done
#


## rerun test random 470-474
#MODE='test'
#METHOD='random'
#BUDGET=2
#for T in 2 4 6 8 10
#do
#    sbatch cpu_run.sh -t $T -b $BUDGET -e $MODE -m $METHOD 
#done

# Feb 25 5pm test random on 500 nodes decrease m from 7 to 6 # 475-479
#MODE='test'
#METHOD='random'
#NODE_TRAIN=500
#NODE_TEST=500
#M=6
#for T in 4 8 12 16 20
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -h $M
#done

# Feb 26 5.10pm test random on 500 nodes decrease p from 0.3 to 0.05; keep m as 7 # 480-484
#MODE='test'
#METHOD='random'
#NODE_TRAIN=500
#NODE_TEST=500
#P=0.05
#for T in 4 8 12 16 20 
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P
#done

#485-489 discard

## Feb 27 2.20pm test random on 500 nodes decrease p from 0.3 to 0.05; decrease m from 7 to 6 # 490-494
#MODE='test'
#METHOD='random'
#NODE_TRAIN=500
#NODE_TEST=500
#P=0.05
#M=6
#for T in 4 8 12 16 20
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P sleep 5
#done
#
## Feb 27 2.20pm train on setting as above # 495-499
#NODE_TRAIN=500
#NODE_TEST=500
#p=0.05
#M=6
#for T in 4 8 12 16 20
#do
#    sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P sleep 5
#done

## Feb 27 2.40pm test random on 500 nodes further decrease m from 6 to 5 # 500-504
#MODE='test'
#METHOD='random'
#NODE_TRAIN=500
#NODE_TEST=500
#P=0.05
#M=5
#for T in 4 8 12 16 20
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P sleep 5
#done
#
## Feb 27 2.20pm train on setting as above # 505-509
#NODE_TRAIN=500
#NODE_TEST=500
#p=0.05
#M=5
#for T in 4 8 12 16 20 
#do
#    sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P sleep 5
#done

# 2.50pm further decrease m to 4  #510-514
MODE='test'
METHOD='random'
NODE_TRAIN=500
NODE_TEST=500
P=0.05
M=4
for T in 4 8 12 16 20
do
    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P sleep 5
done


# Feb 27 2.20pm train on setting as above # 515-519
NODE_TRAIN=500
NODE_TEST=500
p=0.05
M=4
for T in 4 8 12 16 20 
do
    sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P sleep 5
done
















