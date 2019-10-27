#!/usr/bin/env bash

extract_dir="$1"

echo "Downloading MiniSat"
mkdir -p $HOME/.pkgs
cd $HOME/.pkgs
wget http://minisat.se/downloads/minisat-2.2.0.tar.gz

mkdir -p "$extract_dir"
tar xf minisat-2.2.0.tar.gz -C "$extract_dir"
cd "$extract_dir/minisat"

echo "Installing MiniSat"
export MROOT=$(pwd)
cd core
gmake r
ln -sf "$(pwd)/minisat_release" $HOME/local/bin/minisat_core
cd ../simp
gmake r
ln -sf "$(pwd)/minisat_release" $HOME/local/bin/minisat_simp
ln -sf $HOME/local/bin/minisat_core $HOME/local/bin/minisat
