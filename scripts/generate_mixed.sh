#!/usr/bin/env bash

mkdir -p data/mixed

python code/randkcnf.py data/mixed/1 400 3 5 21 0
python code/domset.py data/mixed/1 400 5 0.2 2 400
python code/kclique.py data/mixed/1 400 5 0.2 3 800
python code/kcolor.py data/mixed/1 400 5 0.5 3 1200
python code/kcover.py data/mixed/1 400 5 0.5 2 1600

python code/randkcnf.py data/mixed/2 400 3 10 43 0
python code/domset.py data/mixed/2 400 7 0.2 3 400
python code/kclique.py data/mixed/2 400 10 0.1 3 800
python code/kcolor.py data/mixed/2 400 10 0.5 3 1200
python code/kcover.py data/mixed/2 400 7 0.5 3 1600
