#!/usr/bin/env bash

mkdir -p data/kcolor

python code/kcolor.py data/kcolor/3-5-0.5/ 2000 5 0.5 3 0 &
python code/kcolor.py data/kcolor/3-10-0.5/ 2000 10 0.5 3 0 &
python code/kcolor.py data/kcolor/4-15-0.5/ 2000 15 0.5 4 0 &
python code/kcolor.py data/kcolor/5-20-0.5/ 2000 20 0.5 5 0 &
