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


## Feb 27 4.40pm #510-529
#MODE='test'
#METHOD='random'
#NODE_TRAIN=500
#NODE_TEST=500
#P=0.05
#for M in 3 4 5
#do
#for T in 4 8 12 16 20
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M sleep 10
#done
#done
#
#
#
## Feb 27 2.20pm train on setting as above # 530-539
#NODE_TRAIN=500
#NODE_TEST=500
#p=0.05
#for M in 3 4 5
#do
#for T in 4 8 12 16 20 
#do
#    sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M sleep 10
#done
#done


## Feb 27 11.20pm #545-549
#MODE='test'
#METHOD='random'
#NODE_TRAIN=1000
#NODE_TEST=1000
#P=0.05
#M=6 
#for T in 4 8 12 16 20
#do
#    sleep 3; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M 
#done


# Feb 27 11.40pm increase node size to 10K #
#MODE='test'
#METHOD='random'
#NODE_TRAIN=10000
#NODE_TEST=10000
#P=0.05
#M=6
#for T in 4 8 12 16 20
#do
#    sleep 6; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M
#done

# Feb 28 0.40am increase node size to 1K #
#MODE='test'
#METHOD='random'
#NODE_TRAIN=1000
#NODE_TEST=1000
#P=0.05
#for M in 5 7 
#do
#for T in 4 8 12 16 20
#do
#    sleep 6; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M
#done
#done

# Feb 28 11.50am use T=2 4 6 8 10 12 #570-587
#MODE='test'
#METHOD='random'
#NODE_TRAIN=1000
#NODE_TEST=1000
#BUDGET=2
#P=0.05
#for M in 5 6 7
#do
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET
#done
#done

#588-621 discard


## Feb 28 11.50am use T=2 4 6 8 10 12 # 622-627
#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TRAIN=1000
#NODE_TEST=1000
#BUDGET=2
#P=0.05
#M=6
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET
#done

#628-645 discard

# Feb 28 11.55pm training version of above # 646-651
#NODE_TRAIN=1000
#NODE_TEST=1000
#BUDGET=2
#P=0.05
#M=6
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET
#done
#
#
### Feb 28 11.55pm # rerun lazy-ada-greedy because greedy_sample_size is too large
#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TRAIN=1000
#NODE_TEST=1000
#BUDGET=2
#P=0.05
#M=6
#GREEDY_SAMPLE_SIZE=5
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE
#done

#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TRAIN=1000
#NODE_TEST=1000
#BUDGET=2
#P=0.05
#M=6
#GREEDY_SAMPLE_SIZE=10
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE
#done


## March 1 10.30pm train #683-  
#NODE_TRAIN=500
#NODE_TEST=500
#BUDGET=2
#P=0.05
#M=5
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET
#done
#
## March 1 3pm #
#MODE='test'
#METHOD='random'
#NODE_TEST=500
#BUDGET=2
#P=0.05
#M=5
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET 
#done
#
### March 1 3pm # 707-712
#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TEST=500
#BUDGET=2
#P=0.05
#M=5
#GREEDY_SAMPLE_SIZE=5
#for T in 2 4 6 8 10 12
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE
#done

#695-706 713-723 discard

# Mar 02 0.50am test on erdos renyi graph #724-
MODE='test'
METHOD='random'
NODE_TEST=500
BUDGET=4
GRAPH_TYPE='erdos_renyi'
M=5
P=0.025 
for T in 4 8 12 16 20  
do
    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -y $GRAPH_TYPE
done

MODE='test'
METHOD='lazy_adaptive_greedy'
NODE_TEST=500
BUDGET=4
GRAPH_TYPE='erdos_renyi'
M=5
P=0.025
GREEDY_SAMPLE_SIZE=5
for T in 4 8 12 16 20
do
    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE -y $GRAPH_TYPE
done
#
NODE_TRAIN=500
NODE_TEST=500
BUDGET=4
GRAPH_TYPE='erdos_renyi'
P=0.025
M=5
for T in 4 8 12 16 20
do
    sleep 10; sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -y $GRAPH_TYPE
done





