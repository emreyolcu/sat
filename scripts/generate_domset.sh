#!/usr/bin/env bash

mkdir -p data/domset

python code/domset.py data/domset/2-5-0.2/ 2000 5 0.2 2 0 &
python code/domset.py data/domset/3-7-0.2/ 2000 7 0.2 3 0 &
python code/domset.py data/domset/3-9-0.2/ 2000 9 0.2 3 0 &
python code/domset.py data/domset/4-12-0.2/ 2000 12 0.2 4 0 &
