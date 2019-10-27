#!/usr/bin/env bash

mkdir -p varied

# domset
python code/graph_vary.py varied/domset domset 2 ba --x 10 --y 1
python code/graph_vary.py varied/domset domset 2 rr --x 4 --y 10
python code/graph_vary.py varied/domset domset 2 rg --x 10 --y 0.4
python code/graph_vary.py varied/domset domset 2 ws --x 12 --y 4 --z 0.2

python code/graph_vary.py varied/domset domset 3 ba --x 12 --y 1
python code/graph_vary.py varied/domset domset 3 rr --x 3 --y 12
python code/graph_vary.py varied/domset domset 3 rg --x 12 --y 0.3
python code/graph_vary.py varied/domset domset 3 ws --x 12 --y 3 --z 0.3

python code/graph_vary.py varied/domset domset 4 ba --x 12 --y 1
python code/graph_vary.py varied/domset domset 4 rr --x 3 --y 16
python code/graph_vary.py varied/domset domset 4 rg --x 15 --y 0.25
python code/graph_vary.py varied/domset domset 4 ws --x 14 --y 3 --z 0.3
