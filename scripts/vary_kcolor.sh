#!/usr/bin/env bash

mkdir -p varied

# color
python code/graph_vary.py varied/kcolor kcolor 3 ba --x 15 --y 2
python code/graph_vary.py varied/kcolor kcolor 3 rr --x 4 --y 15
python code/graph_vary.py varied/kcolor kcolor 3 rg --x 15 --y 0.3
python code/graph_vary.py varied/kcolor kcolor 3 ws --x 15 --y 5 --z 0.5

python code/graph_vary.py varied/kcolor kcolor 4 ba --x 15 --y 5
python code/graph_vary.py varied/kcolor kcolor 4 rr --x 7 --y 14
python code/graph_vary.py varied/kcolor kcolor 4 rg --x 15 --y 0.4
python code/graph_vary.py varied/kcolor kcolor 4 ws --x 15 --y 7 --z 0.2

python code/graph_vary.py varied/kcolor kcolor 5 ba --x 15 --y 7
python code/graph_vary.py varied/kcolor kcolor 5 rr --x 9 --y 16
python code/graph_vary.py varied/kcolor kcolor 5 rg --x 15 --y 0.5
python code/graph_vary.py varied/kcolor kcolor 5 ws --x 15 --y 9 --z 0.2
