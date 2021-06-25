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

Then install `pytorch_sparse` via wheel files. Go to this site: https://github.com/rusty1s/pytorch_sparse to download wheel files.

### Docker

Follow the instructions in https://github.com/Haipeng-Chen/RL4IM-with-Contingency/blob/main/docker/README.md


### Run the code
#### RL training
1. Default command line:
```bash
python main.py --config=rl4im --env-config=basic_env --results-dir=results with lr=1e-3
```
All the default environment and method-related parameters are stored in `src/tasks/config`. You can set params after `with`.

2. Alternatively, you may batch run jobs using `sh multi_run.sh`. Example settings in the paper are specified there which could be used to quickly reproduce the results.  

3. Or run with Docker on GPU `0`, for example:

```bash
bash run_interactive.sh 0 python3.7 main.py --config=rl4im --env-config=basic_env --results-dir=results with lr=1e-3
```

#### RL validation (find the best checkpoint)
This step is to find the optimal checkpoint from the validation results. You can use the script validate.ipynb, where you need to specify the runs number. 
We will try to upgrade in the next version with an end-to-end pipeline.


#### RL test

After finding the best check point via the validation process above, you are ready to run the test! An example of RL test is also given in `multi_run.sh`

It will load the model with the max step by default. If you want to load the specified model, add `load_step=OPT`, where OPT is the one you find using the validation process above. 


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


