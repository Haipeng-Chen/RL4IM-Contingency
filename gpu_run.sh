#!/bin/bash
#SBATCH -n 1              # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p tambe_gpu  # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH -o output/%j.out  # File to which STDOUT will be written, %j inserts jobid

T=8
BUDGET=4
SAVE_EVERY=2
Q=0.6
MODE='train'
NODE_TRAIN=200
NODE_TEST=200
M=7
PROPAGATE_P=0.1
P=0.3
METHOD='rl'

while getopts f:g:q:t:b:e:h:i:j:m:s: option
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
e) MODE=${OPTARG};;
h) M=${OPTARG};;
i) PROPAGATE_P=${OPTARG};;
j) P=${OPTARG};;
#methods related
m) METHOD=${OPTARG};;
s) SAVE_EVERY=${OPTARG};;
esac
done

echo 'mode is:' $MODE
echo 'method is:' $METHOD
echo 'budget is:' $BUDGET
#declare -i T

python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with T=$T budget=$BUDGET save_every=$SAVE_EVERY q=$Q mode=$MODE node_train=$NODE_TRAIN node_test=$NODE_TEST m=$M propagate_p=$PROPAGATE_P p=$P method=$METHOD
