#!/usr/bin/env bash

extract_dir="$1"

echo "Downloading Walksat"
mkdir -p "$extract_dir"
git clone https://gitlab.com/HenryKautz/Walksat.git "$extract_dir/Walksat"
cd "$extract_dir/Walksat/Walksat_v56"

echo "Installing Walksat"
make
cp walksat $HOME/local/bin/
cp makewff $HOME/local/bin/
cp makequeens $HOME/local/bin/
cp Scripts/* $HOME/local/bin/
