Default command lines:

```python
python main.py --config=colge --env-config=basic_env --results-dir=temp_dir with lr=1e-4
```

`--config=colge`, `--env-config=basic_env` and `--results-dir=temp_dir` all needed and hyperparameters are stored in `src/tasks/config`. You can set params after `with`.


Other commands:
```python
python -m src.environment.env --baseline 'random' --cascade 'DIC'
```