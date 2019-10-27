#!/usr/bin/env bash

mkdir -p varied

# cover
python code/graph_vary.py varied/kcover kcover 4 ba --x 10 --y 2
python code/graph_vary.py varied/kcover kcover 4 rr --x 3 --y 8
python code/graph_vary.py varied/kcover kcover 4 rg --x 10 --y 0.3
python code/graph_vary.py varied/kcover kcover 4 ws --x 10 --y 2 --z 0.2

python code/graph_vary.py varied/kcover kcover 5 ba --x 10 --y 3
python code/graph_vary.py varied/kcover kcover 5 rr --x 3 --y 10
python code/graph_vary.py varied/kcover kcover 5 rg --x 12 --y 0.3
python code/graph_vary.py varied/kcover kcover 5 ws --x 12 --y 3 --z 0.3

python code/graph_vary.py varied/kcover kcover 6 ba --x 12 --y 4
python code/graph_vary.py varied/kcover kcover 6 rr --x 3 --y 12
python code/graph_vary.py varied/kcover kcover 6 rg --x 15 --y 0.25
python code/graph_vary.py varied/kcover kcover 6 ws --x 14 --y 3 --z 0.2
