#!/bin/bash

input="12matrices.csv"
{
  read
  i=0
  while IFS=',' read -r mid group name rows cols nonzeros
  do
    echo "$mid $group $name $rows $cols $nonzeros"
    echo "~/matrix/$group/$name/$name.mtx"
    ./../bfs/build/bin/bfs_SELLCSv2_DFC_omp -m ../MM/$group/$name/$name.cbd -p 0
    i=`expr $i + 1`
  done 
} < "$input"
