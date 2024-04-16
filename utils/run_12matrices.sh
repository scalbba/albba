#!/bin/bash


#installation and compilation
mkdir ../bfs/build
cd ../bfs/build
../configure llvm_ncc
make
cd ..

#data preparation
python3 download_12matrices.py

#experimental evaluation--performance data collection
bash bench-run12matrices.sh

# experimental evaluation--figure plotting
cd plot
python3 perf-bar.py