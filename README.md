Code for the paper :
## "Contingency-Aware Influence Maximization: A Reinforcement Learning Approach"

#### Authors : Haipeng Chen<sup>1</sup>, Wei Qiu<sup>2</sup>, Han-Ching Ou<sup>1</sup>, Bo An<sup>2</sup>, Milind Tambe<sup>1</sup>
#### <sup>1</sup>Harvard University &nbsp; &nbsp; <sup>2</sup>Nanyang Technological University

#### Accepted by UAI-2021.


## Instructions:
### Installation
Install `pytorch_sparse` via wheel files go to this site: https://github.com/rusty1s/pytorch_sparse


### Run the code
#### RL training
1. Default command line:
```python
python main.py --config=rl4im --env-config=basic_env --results-dir=results with lr=1e-3
```
All the default environment and method-related parameters are stored in `src/tasks/config`. You can set params after `with`.

2. Alternatively, you may batch run jobs using `sh multi_run.sh`. Example settings in the paper are specified there which could be used to quickly reproduce the results.  

Run with Docker on GPU `0`

```bash
bash run_interactive.sh 0 python3.7 main.py --config=rl4im --env-config=basic_env --results-dir=temp_dir with lr=1e-4
```

#### RL validation (find best checkpoint)
Unfortunately we are doing it separately with a notebook script at the current stage. We will try to upgrade in the next version with an end-to-end pipeline.

Currently, you will need to run a script `xxx` to return the optimal checkpoint.

#### RL test

After finding the best check point via the validation process above, you are ready to run the test! An example of RL test is also given in `multi_run.sh`

It will load the model with the max step. If you want to load the specified model, add `load_step=600`.


#### Run CHANGE baseline
An example code of it is also included in `multi_run.sh`

### Cite

