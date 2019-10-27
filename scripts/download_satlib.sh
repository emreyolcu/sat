#!/usr/bin/env bash

extract_dir="$1"

echo "Changing directory to $extract_dir"
mkdir -p "$extract_dir"
cd "$extract_dir"

uf_sets=("50-218" "75-325" "100-430" "125-538" "150-645" "175-753" "200-860" "225-960" "250-1065")
flat_sets=("30-60" "50-115" "75-180" "100-239" "125-301" "150-360" "175-417" "200-479")
sw_sets=("0" "1" "2" "3" "4" "5" "6" "7" "8")

extract_set() {
    mkdir -p "$1"
    tar xvf "$1.tar.gz" -C "$1" --transform="s#^.+/##x"
    rm "$1.tar.gz"
}

mkdir -p "uf"
cd "uf"
echo "Downloading Uniform Random-3-SAT"
for s in ${uf_sets[*]}; do
    set_name="uf$s"
    wget "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/$set_name.tar.gz"
    wget "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/u$set_name.tar.gz"
    extract_set $set_name
    extract_set "u${set_name}"
done
cd ..

mkdir -p "flat"
cd "flat"
echo "Downloading Flat Graph Coloring"
for s in ${flat_sets[*]}; do
    set_name="flat$s"
    wget "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/$set_name.tar.gz"
    extract_set $set_name
done
cd ..

mkdir -p "sw"
cd "sw"
echo "Downloading Morphed Graph Coloring"
for s in ${sw_sets[*]}; do
    set_name="sw100-8-lp${s}-c5"
    wget "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/$set_name.tar.gz"
    extract_set $set_name
done
cd ..
