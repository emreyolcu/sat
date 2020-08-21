# Learning local search heuristics for SAT
Code accompanying the NeurIPS 2019 paper [*Learning local search heuristics for Boolean satisfiability*](https://papers.nips.cc/paper/9012-learning-local-search-heuristics-for-boolean-satisfiability).

## Setup
Make sure to have `minisat` in your path. It is available here: http://minisat.se/MiniSat.html

Run the following sequence of commands to clone the repository and create the required conda environments.

```
git clone --recursive https://github.com/emreyolcu/sat.git
cd sat
conda env create -f sat.yaml
conda env create -f cnfgen.yaml
(cd code/PyMiniSolvers && make)
```

## Instructions
### Data generation
`scripts` directory includes bash scripts to generate the data used for training. As an example, to generate the coloring formulas, run the following commands.

```
conda activate cnfgen
scripts/generate_kcolor.sh
```

When generating the SR formulas, activate the `sat` environment instead of `cnfgen` before running the script.

### Training
`configs/search` directory includes training configurations for each set of formulas. Run the following commands to train models on sets of coloring formulas of increasing size.

```
conda activate sat
python code/train_search.py --config_path configs/search/kcolor/5.yaml
python code/train_search.py --config_path configs/search/kcolor/10.yaml
python code/train_search.py --config_path configs/search/kcolor/15.yaml
python code/train_search.py --config_path configs/search/kcolor/20.yaml
```

If a configuration file has the line `eval_multi: True`, the training script will create 25 processes to perform evaluation during training. Remove this line or change the value to `False` if you prefer to use a single process instead.

When attempting to use multiple processes you may see the error `Too many open files`. This can be avoided by executing `ulimit -n 4096` (or another large value that your system allows) before starting training.

### Evaluation
After training is completed, run the following commands to evaluate a learned model and WalkSAT on a set of formulas.

```
conda activate sat
python code/evaluate.py --dir_path data/kcolor/3-10-0.5 --samples 5 --model_path results/search/kcolor/5/model_best.pth --no_multi
python code/walksat.py --dir_path data/kcolor/3-10-0.5 --samples 5
```

Arguments used:
- `--dir_path`: Directory containing the formulas.
- `--samples`: Number of formulas to run the solver on.
- `--model_path`: Saved parameters of the graph neural network to evaluate.
- `--no_multi`: Whether to use a single process for evaluation. Without this argument, the script creates 25 processes.
