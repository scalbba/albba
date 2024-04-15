#!/bin/bash

# Aurora compilers and tools
export PATH=/opt/nec/ve/bin:$PATH

# NEC VE NLC env vars
# Note: this enables the use of the NEC MATH LIBRARY (NLC)
source /opt/nec/ve/nlc/2.0.0/bin/nlcvars.sh i64
export NCC_INCLUDE_PATH=$NCC_INCLUDE_PATH:/opt/nec/ve/nlc/2.0.0/include/inc_i64
export VE_LIBRARY_PATH=$VE_LIBRARY_PATH:/opt/nec/ve/nlc/2.0.0/lib


export PATH=/opt/nec/nosupport/llvm-ve/bin:$PATH

export VE_NODE_NUMBER=0
export OMP_NUM_THREADS=1
echo "Commands will be executed in VE#"${VE_NODE_NUMBER}
echo "OMP_NUM_THREADS="${OMP_NUM_THREADS}
