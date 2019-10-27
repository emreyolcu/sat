#!/usr/bin/env bash

mkdir -p data/kclique

python code/kclique.py data/kclique/5-0.2/ 2000 5 0.2 3 0 &
python code/kclique.py data/kclique/10-0.1/ 2000 10 0.1 3 0 &
python code/kclique.py data/kclique/15-0.066/ 2000 15 0.066 3 0 &
python code/kclique.py data/kclique/20-0.05/ 2000 20 0.05 3 0 &
