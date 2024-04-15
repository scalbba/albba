#!/bin/bash


#installation and compilation
mkdir ../bfs/build
cd ../bfs/build
../configure llvm_ncc
make
cd ..

#data preparation
python3 download_dataset.py

#experimental evaluation--performance data collection
bash bench-runtime.sh

experimental evaluation--figure plotting
cd plot
python3 perf-scatter-hits.py
python3 bypass-hitsgram.py
