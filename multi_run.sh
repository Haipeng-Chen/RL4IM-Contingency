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
### March 1 3pm # 707-712 # none of the following finishes even for one graph
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

## Mar 02 0.50am test on erdos renyi graph #724-728
#MODE='test'
#METHOD='random'
#NODE_TEST=500
#BUDGET=4
#GRAPH_TYPE='erdos_renyi'
#M=5
#P=0.025 
#for T in 4 8 12 16 20  
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -y $GRAPH_TYPE
#done
#
##729-733
#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TEST=500
#BUDGET=4
#GRAPH_TYPE='erdos_renyi'
#M=5
#P=0.025
#GREEDY_SAMPLE_SIZE=5
#for T in 4 8 12 16 20
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE -y $GRAPH_TYPE
#done
#
## 734-738
#NODE_TRAIN=500
#NODE_TEST=500
#BUDGET=4
#GRAPH_TYPE='erdos_renyi'
#P=0.025
#M=5
#for T in 4 8 12 16 20
#do
#    sleep 10; sbatch gpu_run.sh -t $T -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -y $GRAPH_TYPE
#done



# March 20 run DIC model # 746-755
#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='DIC'
#for GREEDY_SAMPLE_SIZE in 5 10 20 50 100
#do
#for T in 6 8
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE -k $CASCADE
#done
#done

# March 20 # 743 740
#MODE='test'
#METHOD='random'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#GREEDY_SAMPLE_SIZE=5
#CASCADE='DIC'
#for T in 6 8
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE -k $CASCADE
#done

# March 20 10.15pm # 756-767
#MODE='train'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#NUM_SIMUL_TRAIN=200
#CASCADE='DIC'
#for LR in 0.0001 0.01
#do
#for EPSILON_DECAY_STEPS in 1000 1500
#do
#for T in 6 
#do
#    sleep 10; sbatch gpu_run.sh -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -y $GRAPH_TYPE -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN
#done
#done
#done

# March 21 0.40am # 772-776
#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='DIC'
#GREEDY_SAMPLE_SIZE=50
#for T in 4 6 8 10 12
#do
#    sleep 10; sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE -k $CASCADE
#done

# March 21 0.40am #777-796
#MODE='train'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#NUM_SIMUL_TRAIN=200
#CASCADE='DIC'
#GRAPH_TYPE='powerlaw'
#T=6
#EPSILON_DECAY_STEPS=1500
#LR=0.001
#for NUM_SIMUL_TRAIN in 1 10 50 100 200 
#do
#for ITER  in 1 2 3 4 # see if RL is stable 
#do
#    sleep 10; sbatch gpu_run.sh -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN
#done
#done

# March 21 10.40pm train RL4IM # 797-811
#MODE='train'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#NUM_SIMUL_TRAIN=10
#CASCADE='DIC'
#GRAPH_TYPE='powerlaw'
#T=6
#EPSILON_DECAY_STEPS=1500
#LR=0.001
#for T in 4 6 8 10 12
#do
#for ITER in 1 2 3 
#do 
#    sbatch gpu_run.sh -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN; sleep 20
#done
#done


# 812, 818-822
#MODE='test'
#METHOD='rl'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='DIC'
#GREEDY_SAMPLE_SIZE=50
#GRAPH_TYPE='powerlaw'
##
##T=6
##CHECK_POINT_PATH=./temp_dir/colge/sacred/783/models
##LOAD_STEP=1704
##sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20
## 
#T=4
#CHECK_POINT_PATH=./temp_dir/colge/sacred/799/models
#LOAD_STEP=1100
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=6
#CHECK_POINT_PATH=./temp_dir/colge/sacred/801/models
#LOAD_STEP=1524
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=8
#CHECK_POINT_PATH=./temp_dir/colge/sacred/803/models
#LOAD_STEP=1640
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=10
#CHECK_POINT_PATH=./temp_dir/colge/sacred/807/models
#LOAD_STEP=1620
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=12
#CHECK_POINT_PATH=./temp_dir/colge/sacred/811/models
#LOAD_STEP=1608
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;


# March 21 random # 813-817
#MODE='test'
#METHOD='random'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='DIC'
#GREEDY_SAMPLE_SIZE=50
#for T in 4 6 8 10 12
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -z $GREEDY_SAMPLE_SIZE -k $CASCADE; sleep 20
#done

# Mar 22 3.30pm test CHANGE on IC model vary sample size # 823-827 (T=8), 843-847 (T=6) 
#MODE='test'
#METHOD='lazy_adaptive_greedy'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='IC'
#GRAPH_TYPE='powerlaw'
#for GREEDY_SAMPLE_SIZE in 5 10 20 50 100
#do
#for T in 6
#do 
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE; sleep 20;
#done
#done

# Mar 22 3.40pm train RL on IC model # 828-842
#MODE='train'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#NUM_SIMUL_TRAIN=10
#CASCADE='IC'
#GRAPH_TYPE='powerlaw'
#EPSILON_DECAY_STEPS=1500
#LR=0.001
#for T in 4 6 8 10 12
#do
#for ITER in 1 2 3 
#do 
#   sbatch gpu_run.sh -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN; sleep 20;
#done
#done

# test RL on IC model #848-852 
#MODE='test'
#METHOD='rl'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='IC'
#GREEDY_SAMPLE_SIZE=50
#GRAPH_TYPE='powerlaw'
#T=4
#CHECK_POINT_PATH=./temp_dir/colge/sacred/828/models
#LOAD_STEP=1720
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=6
#CHECK_POINT_PATH=./temp_dir/colge/sacred/832/models
#LOAD_STEP=1242
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=8
#CHECK_POINT_PATH=./temp_dir/colge/sacred/836/models
#LOAD_STEP=1840
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=10
#CHECK_POINT_PATH=./temp_dir/colge/sacred/837/models
#LOAD_STEP=1440
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=12
#CHECK_POINT_PATH=./temp_dir/colge/sacred/840/models
#LOAD_STEP=1560
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;


# test random/CHANGE (sample = 50 5) on IC model #854-868
#MODE='test'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='IC'
#GRAPH_TYPE='powerlaw'
#GREEDY_SAMPLE_SIZE=5
#METHOD='lazy_adaptive_greedy'
#for T in 4 6 8 10 12
#do 
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE; sleep 20;
#done

# previous DIC model runs are wrong; test random, CHANGE (sample=5, 50) on DIC model # propagation_p=0.1: 904-938; 0.2: 939-968 (for all the following 4 set of exps) 
#MODE='test'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#CASCADE='DIC'
#PROPAGATE_P=0.2
#GRAPH_TYPE='powerlaw'
#GREEDY_SAMPLE_SIZE=5
#METHOD='lazy_adaptive_greedy'
#for GREEDY_SAMPLE_SIZE in 5 50
#do
#for T in 4 8 12 16 20
#do
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE; sleep 20;
#done
#done

## test random on DIC model 
#MODE='test'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#PROPAGATE_P=0.2
#CASCADE='DIC'
#GRAPH_TYPE='powerlaw'
#GREEDY_SAMPLE_SIZE=5
#METHOD='random'
#for T in 4 8 12 16 20 
#do 
#    sbatch cpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE; sleep 20;
#done
#
## train RL DIC model 
#MODE='train'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#PROPAGATE_P=0.2
#NUM_SIMUL_TRAIN=10
#CASCADE='DIC'
#GRAPH_TYPE='powerlaw'
#T=6
#EPSILON_DECAY_STEPS=1500
#LR=0.001
#for T in 4 8 12 16 20 
#do
#for ITER in 1 2 3 
#do 
#    sbatch gpu_run.sh -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN; sleep 20
#done
#done

# RL test DIC model 
#MODE='test'
#METHOD='rl'
#NODE_TEST=200
#BUDGET=2
#P=0.05
#M=5
#PROPAGATE_P=0.2
#CASCADE='DIC'
#GREEDY_SAMPLE_SIZE=50
#GRAPH_TYPE='powerlaw'
#T=4
#CHECK_POINT_PATH=./temp_dir/colge/sacred/920/models
#LOAD_STEP=1900
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=8
#CHECK_POINT_PATH=./temp_dir/colge/sacred/923/models
#LOAD_STEP=1880
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=12
#CHECK_POINT_PATH=./temp_dir/colge/sacred/926/models
#LOAD_STEP=1860
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=16
#CHECK_POINT_PATH=./temp_dir/colge/sacred/930/models
#LOAD_STEP=1968
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
#T=20
#CHECK_POINT_PATH=./temp_dir/colge/sacred/931/models
#LOAD_STEP=1740
#sbatch gpu_run.sh -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -z $GREEDY_SAMPLE_SIZE -k $CASCADE -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;

#----------------------------------March 28: Experiments from now are recorded in results/rl4im------------------------
## ER: 75-84 #PL: 90-99 
#Q=1.0
#MODE='test'
#NODE_TEST=200
#BUDGET=8
#P=0.05
#M=4
#PROPAGATE_P=0.1
#CASCADE='DIC'
##GRAPH_TYPE='erdos_renyi'
#GRAPH_TYPE='powerlaw'
#GREEDY_SAMPLE_SIZE=50
#METHOD='lazy_adaptive_greedy' 
#IS_REAL_GRAPH=False
#for METHOD in 'lazy_adaptive_greedy' 'random'
#do
#for T in 4 8 12 16 20 
#do  
#    BUDGET=$T; sbatch cpu_run.sh -q $Q -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -z $GREEDY_SAMPLE_SIZE -k $CASCADE; sleep 20;
#done
#done

## train RL DIC model #ER: 60-74 #PL: 100-1014
#Q=1.0
#MODE='train'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=1
#P=0.05
#M=4
#PROPAGATE_P=0.1
#NUM_SIMUL_TRAIN=50
#CASCADE='DIC'
##GRAPH_TYPE='erdos_renyi'
#GRAPH_TYPE='powerlaw'
#IS_REAL_GRAPH=False
#T=8
#EPSILON_DECAY_STEPS=1500
#LR=0.003
#for T in 4 8 12 16 20 
#do
#BUDGET=$T
#for ITER in 1 2 3 
#do 
#    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN; sleep 20;
#done
#done


# RL test #ER: 85-89 #PL: 115-119
#Q=1.0
#MODE='test'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=1
#P=0.05
#M=4
#PROPAGATE_P=0.1
#NUM_SIMUL_TRAIN=50
#CASCADE='DIC'
##GRAPH_TYPE='erdos_renyi'
#GRAPH_TYPE='powerlaw'
#IS_REAL_GRAPH=False
#T=8
#EPSILON_DECAY_STEPS=1500
#LR=0.003
## ER graph: 85-89
#T=4
#CHECK_POINT_PATH=./results/rl4im/sacred/61/models
#LOAD_STEP=1640
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=8
#CHECK_POINT_PATH=./results/rl4im/sacred/65/models
#LOAD_STEP=640
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=12
#CHECK_POINT_PATH=./results/rl4im/sacred/66/models
#LOAD_STEP=1860
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=16
#CHECK_POINT_PATH=./results/rl4im/sacred/71/models
#LOAD_STEP=1312
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=20
#CHECK_POINT_PATH=./results/rl4im/sacred/74/models
#LOAD_STEP=800
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
## PL graph: 115-119
#T=4
#CHECK_POINT_PATH=./results/rl4im/sacred/100/models
#LOAD_STEP=1380
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=8
#CHECK_POINT_PATH=./results/rl4im/sacred/105/models
#LOAD_STEP=1104
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=12
#CHECK_POINT_PATH=./results/rl4im/sacred/106/models
#LOAD_STEP=780
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=16
#CHECK_POINT_PATH=./results/rl4im/sacred/109/models
#LOAD_STEP=1984
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=20
#CHECK_POINT_PATH=./results/rl4im/sacred/113/models
#LOAD_STEP=1580
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;

# 120-145 are random trials

#-------------------------------- March 30: below are trials for real world graph -----------------------------------------
#146-155
#Q=1.0
#MODE='test'
#NODE_TEST=200
#BUDGET=8
#P=0.05
#M=4
#PROPAGATE_P=0.1
#CASCADE='DIC'
#GRAPH_TYPE='powerlaw'
#GREEDY_SAMPLE_SIZE=50
#METHOD='lazy_adaptive_greedy' 
#IS_REAL_GRAPH=True
#SAMPLE_NODES_RATIO=0.8
#REAL_GRAPH_NAME='polbooks'
#for METHOD in 'lazy_adaptive_greedy' 'random'
#do
#for T in 4 8 12 16 20 
#do  
#    BUDGET=$T; sbatch cpu_run.sh -q $Q -t $T -e $MODE -m $METHOD -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -z $GREEDY_SAMPLE_SIZE -k $CASCADE; sleep 20;
#done
#done

# 156-168 are incorrect runs

# train RL on real graph DIC model # 169-183
#Q=1.0
#MODE='train'
#NODE_TRAIN=200
#NODE_TEST=200
#BUDGET=1
#P=0.05
#M=4
#PROPAGATE_P=0.1
#NUM_SIMUL_TRAIN=50
#CASCADE='DIC'
#GRAPH_TYPE='powerlaw'
#IS_REAL_GRAPH=True
#SAMPLE_NODES_RATIO=0.8
#REAL_GRAPH_NAME='polbooks'
#T=8
#EPSILON_DECAY_STEPS=1500
#LR=0.003
##for T in 4 8 12 16 20 
##do
##BUDGET=$T
##for ITER in 1 2 3 
##do
##    sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN; sleep 20;
##done
##done
## test RL sample_nodes_ratio=0.8 #209-214
#T=4
#CHECK_POINT_PATH=./results/rl4im/sacred/196/models
#LOAD_STEP=940
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=8
#CHECK_POINT_PATH=./results/rl4im/sacred/199/models
#LOAD_STEP=1120
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=12
#CHECK_POINT_PATH=./results/rl4im/sacred/202/models
#LOAD_STEP=1080
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=16
#CHECK_POINT_PATH=./results/rl4im/sacred/205/models
#LOAD_STEP=1968
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;
##
#T=20
#CHECK_POINT_PATH=./results/rl4im/sacred/206/models
#LOAD_STEP=1500
#BUDGET=$T
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -v $REAL_GRAPH_NAME -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN -p $CHECK_POINT_PATH -r $LOAD_STEP; sleep 20;

# April 15
# train RL on IC model with 1e3 nodes and validate on  1e4 nodes graphs
#MODE='train'
#NODE_TRAIN=1000
#NODE_TEST=10000
#BUDGET=2
#Q=0.6
#P=0.05
#M=3
#PROPAGATE_P=0.1
#NUM_SIMUL_TRAIN=10
#CASCADE='IC'
#IS_REAL_GRAPH=False
#GRAPH_TYPE='powerlaw'
#EPSILON_DECAY_STEPS=1500
#LR=0.001
#T=8 
#for ITER in 1 2 3 
#do 
#   sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN; sleep 20;
#done

# test random on 1e4 nodes graphs
#MODE='test'
#NODE_TRAIN=1000
#NODE_TEST=10000
#BUDGET=2
#Q=0.6
#P=0.05
#M=3
#PROPAGATE_P=0.1
#NUM_SIMUL_TRAIN=10
#CASCADE='IC'
#METHOD='random' 
#IS_REAL_GRAPH=False
#SAMPLE_NODES_RATIO=0.8
#REAL_GRAPH_NAME='India'
#GRAPH_TYPE='powerlaw'
#EPSILON_DECAY_STEPS=1500
#LR=0.001
#T=8
#sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -o $NUM_SIMUL_TRAIN; sleep 20;

# train RL on IC model on real graphs 
MODE='train'
NODE_TRAIN=200
NODE_TEST=200
BUDGET=2
Q=0.6
P=0.05
M=3
PROPAGATE_P=0.1
NUM_SIMUL_TRAIN=10
CASCADE='IC'
METHOD='rl' 
IS_REAL_GRAPH=True
SAMPLE_NODES_RATIO=0.95
REAL_GRAPH_NAME='India'
GRAPH_TYPE='powerlaw'
EPSILON_DECAY_STEPS=1500
LR=0.003
T=8 
for SAMPLE_NODES_RATIO in 0.9 0.95
do
for ITER in 1 2 3 
do 
   sbatch gpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -m $METHOD -o $NUM_SIMUL_TRAIN; sleep 20;
done
done

MODE='test'
NODE_TRAIN=200
NODE_TEST=200
BUDGET=2
Q=0.6
P=0.05
M=3
PROPAGATE_P=0.1
NUM_SIMUL_TRAIN=10
CASCADE='IC'
METHOD='lazy_adaptive_greedy' 
IS_REAL_GRAPH=True
SAMPLE_NODES_RATIO=0.95
REAL_GRAPH_NAME='India'
GRAPH_TYPE='powerlaw'
EPSILON_DECAY_STEPS=1500
LR=0.003
T=8
for METHOD in 'lazy_adaptive_greedy' 'random'
do
for SAMPLE_NODES_RATIO in 0.9 0.95
do
    sbatch cpu_run.sh -q $Q -t $T -e $MODE -f $NODE_TRAIN -g $NODE_TEST -j $P -h $M -i $PROPAGATE_P -b $BUDGET -c $GRAPH_TYPE -d $IS_REAL_GRAPH -u $SAMPLE_NODES_RATIO -k $CASCADE -n $EPSILON_DECAY_STEPS -l $LR -m $METHOD -o $NUM_SIMUL_TRAIN; sleep 20;
done
done
