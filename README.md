Default command lines:

```python
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-4
```

`--config=colge`, `--env-config=basic_env` and `--results-dir=temp_dir` all needed and hyperparameters are stored in `src/tasks/config`. You can set params after `with`.
Default parameter values are in network-opt-rl/src/tasks/config/

Run with Docker on GPU `0`

```bash
bash run_interactive.sh 0 python3.7 main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-4
```

Other commands:
```python
python -m src.environment.env --baseline 'random' --cascade 'DIC'
```

Load Models
```
python3.7 main.py --config=colge --env-config=basic_env --results-dir=results/tmp with lr=1e-2 T=6 checkpoint_path=./results/tmp/colge/sacred/2/models
```

It will load the model with the max step. If you want to load the specified model, add `load_step=600`.

Test

run the `IC.py` in home dir via 

```
python3.7 -m src.IC 
```


Misc.

Install `pytorch_sparse` via wheel files go to this site: https://github.com/rusty1s/pytorch_sparse
