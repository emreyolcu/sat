#!/usr/bin/env bash

mkdir -p varied

# clique
python code/graph_vary.py varied/kclique kclique 3 ba --x 15 --y 2
python code/graph_vary.py varied/kclique kclique 3 rr --x 2 --y 15
python code/graph_vary.py varied/kclique kclique 3 rg --x 15 --y 0.1
python code/graph_vary.py varied/kclique kclique 3 ws --x 15 --y 2 --z 0.2

python code/graph_vary.py varied/kclique kclique 4 ba --x 15 --y 3
python code/graph_vary.py varied/kclique kclique 4 rr --x 4 --y 15
python code/graph_vary.py varied/kclique kclique 4 rg --x 15 --y 0.2
python code/graph_vary.py varied/kclique kclique 4 ws --x 15 --y 4 --z 0.2

python code/graph_vary.py varied/kclique kclique 5 ba --x 15 --y 4
python code/graph_vary.py varied/kclique kclique 5 rr --x 8 --y 15
python code/graph_vary.py varied/kclique kclique 5 rg --x 15 --y 0.3
python code/graph_vary.py varied/kclique kclique 5 ws --x 15 --y 6 --z 0.2
