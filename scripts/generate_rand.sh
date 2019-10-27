#!/usr/bin/env bash

mkdir -p data/rand/

python code/randkcnf.py data/rand/5-21/ 1000 3 5 21 0 &
python code/randkcnf.py data/rand/10-43/ 1000 3 10 43 0 &
python code/randkcnf.py data/rand/25-106/ 1000 3 25 106 0 &
python code/randkcnf.py data/rand/50-213/ 1000 3 50 213 0 &
python code/randkcnf.py data/rand/75-320/ 1000 3 75 320 0 &
python code/randkcnf.py data/rand/100-426/ 1000 3 100 426 0 &
python code/randkcnf.py data/rand/150-639/ 1000 3 150 639 0 &
python code/randkcnf.py data/rand/200-852/ 1000 3 200 852 0 &
