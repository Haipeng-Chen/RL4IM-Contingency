Code for the paper :
## "Contingency-Aware Influence Maximization: A Reinforcement Learning Approach"

#### Authors : Haipeng Chen<sup>1</sup>, Wei Qiu<sup>2</sup>, Han-Ching Ou<sup>1</sup>, Bo An<sup>2</sup>, Milind Tambe<sup>1</sup>
#### <sup>1</sup>Harvard University &nbsp; &nbsp; <sup>2</sup>Nanyang Technological University

#### Accepted by UAI-2021.


## Instructions:
### Installation

First, install libraries via `requirements.txt`:
```
pip install -r requirements.txt 
```

Then install `pytorch_sparse` via wheel files. Go to this site: https://github.com/rusty1s/pytorch_sparse to install or download wheel files to speedup installation via this link: https://pytorch-geometric.com/whl/.


### Run the code
#### RL training
1. Default command line:
```bash
python main.py --config=rl4im --env-config=basic_env --results-dir=results with lr=1e-3
```
All the default environment and method-related parameters are stored in `src/tasks/config`. You can set customized values of hyperparameters after `with` as demonstrated in the above command.

2. There are three ways to run tasks, running tasks with bash scripts on the current machine, running tasks on a distributed system via SLURM or running tasks with Docker. You can run jobs using `sh multi_run.sh`. Example settings in the paper are specified there which could be used to reproduce the results.

(1) Running tasks with bash scripts on the current machine. For example running tasks on GPU `0`:

```bash
bash multi_run.sh --platform normal --gpuid 0
```
The `normal` means running tasks in the Python environment of current machine.

(2) Running tasks on a distributed system via SLURM. For example running tasks via SLURM on GPU `0`:

```bash
bash multi_run.sh --platform slurm --gpuid 0
```

(3) Running tasks with Docker. Follow the instructions in https://github.com/Haipeng-Chen/RL4IM-with-Contingency/blob/main/docker/README.md For example running tasks with Docker on GPU `0`:

```bash
bash multi_run.sh --platform docker --gpuid 0
```

#### RL validation (find the best checkpoint)
The training results will be stored under the directory `results/rl4im/sacred/xx` where `xx` is the id of one running of the experiment.
This step is to find the optimal checkpoint from the validation results. You can use the script `validation.ipynb`, where you need to specify the runs id `xx`. We will try to upgrade in the next version with an end-to-end pipeline.


#### RL test

After finding the best check point via the validation process above, you are ready to run the test! An example of RL test is also given in `multi_run.sh`

It will load the model with the max step by default. If you want to load the specified model, add `load_step=#OPT`, where `#OPT` is the optimal checkpoint you find using the validation process above. 


#### Run CHANGE baseline
An example code of it is also included in `multi_run.sh`

### Cite

```
@article{chen2021contingency,
  title={Contingency-Aware Influence Maximization: A Reinforcement Learning Approach},
  author={Chen, Haipeng and Qiu, Wei and Ou, Han-Ching and An, Bo and Tambe, Milind},
  journal={arXiv preprint arXiv:2106.07039},
  year={2021}
}
```


