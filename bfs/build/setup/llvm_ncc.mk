# - BFS Directory Structure / BFS library 
# ----------------------------------------------------------------------

TOPdir       = ..
SRCdir       = $(TOPdir)/src
INCdir       = $(TOPdir)/src
BINdir       = $(TOPdir)/bin


# - BFS includes / libraries / specifics 
# ----------------------------------------------------------------------

#NLC_LIB = -lsblas_openmp_i64
#NLC_INC = #-I/opt/nec/ve/nlc/2.0.0/include/inc_i64

BFS_INCLUDES = -I/usr/local/ve/llvm-ve-rv-2.2.0/lib/clang/14.0.0/include -I$(INCdir) # $(NLC_INC)
BFS_LIBS     = -lm -static # $(NLC_LIB)

# - Compile time options 
# ----------------------------------------------------------------------
# -DSPMV_VERIFY Define to enable SPMV results verification
# -DUSE_OMP      Enables OpenMP parallel regions in the code
# -DUSE_OMPSS2   Enables OmpSS-2 parallel regions in the code (alternative to DUSE_OMP)
#
# By default SPMV will:
#    *) Not verify results

# SPMV_OPTS     = -DSPMV_VERIFY -DALIGN_TO=2048 -DUSE_OMP # -DENABLE_HWC_VE -DSPMV_DEBUG
 BFS_OPTS     = -DSPMV_VERIFY -DALIGN_TO=2048 -DUSE_OMP -DENABLE_DFC #-DENABLE_NC #-DENABLE_HWC_VE 
#SPMV_OPTS     = -DSPMV_VERIFY -DALIGN_TO=2048 -DUSE_OMP -DENABLE_DFC  #-DENABLE_HWC_VE 
# SPMV_OPTS     = -DSPMV_VERIFY -DALIGN_TO=2048 -DUSE_OMP -DENABLE_NC  #-DENABLE_HWC_VE 

# ----------------------------------------------------------------------
BIN_SUFFIX=

ifneq (,$(findstring ENABLE_NC,$(BFS_OPTS)))
    BIN_SUFFIX :=$(BIN_SUFFIX)_NC
endif

ifneq (,$(findstring ENABLE_DFC,$(BFS_OPTS)))
    BIN_SUFFIX :=$(BIN_SUFFIX)_DFC
endif

ifneq (,$(findstring USE_OMP,$(BFS_OPTS)))
    BIN_SUFFIX :=$(BIN_SUFFIX)_omp
endif

ifneq (,$(findstring ENABLE_HWC_VE,$(BFS_OPTS)))
    BIN_SUFFIX :=$(BIN_SUFFIX)_hwc
endif

# ----------------------------------------------------------------------

BFS_DEFS     = $(BFS_OPTS) $(BFS_INCLUDES)

# ----------------------------------------------------------------------

TARGETS        =   SELLCSv2_par #SELLCS_blocked_advanced_par
# TARGETS        =  SELLCS_blocked_advanced_par  

# - Compilers / linkers - Optimization flags
# ----------------------------------------------------------------------

CC          = ncc
LLVM        = clang
LLVM_FLAGS  += --target=ve-linux $(BFS_DEFS) -O3 -std=c11 -fzvector
CFLAGS      += $(BFS_DEFS) -O3 -Wall -fopenmp # -std=c11

#LLVM_FLAGS  += --target=ve-linux $(BFS_DEFS) -g -ggdb  -O0 -std=c11 -fzvector
#CFLAGS      += $(BFS_DEFS) -O0 -g -Wall -fopenmp 

MASM        = 

LINKER       = $(CC)
LINKFLAGS    = $(CFLAGS)

ARCHIVER     = ar
ARFLAGS      = r
RANLIB       = echo



