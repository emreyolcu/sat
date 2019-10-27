#!/usr/bin/env bash

mkdir -p data/sr/

python code/gen_sr_dimacs.py data/sr/5 1000 --min_n 5 --max_n 5 &
python code/gen_sr_dimacs.py data/sr/10 1000 --min_n 10 --max_n 10 &
python code/gen_sr_dimacs.py data/sr/25 1000 --min_n 25 --max_n 25 &
python code/gen_sr_dimacs.py data/sr/50 1000 --min_n 50 --max_n 50 &
python code/gen_sr_dimacs.py data/sr/100 1000 --min_n 100 --max_n 100 &
python code/gen_sr_dimacs.py data/sr/150 1000 --min_n 150 --max_n 150 &
python code/gen_sr_dimacs.py data/sr/200 1000 --min_n 200 --max_n 200 &
