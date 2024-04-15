# by default, "arch" is llvm_ncc
arch ?= llvm_ncc
SRC_PATH = BFS_SRC_PATH

setup_file = setup/$(arch).mk
include $(setup_file)

COMMON_SRC = 	$(SRCdir)/main.c \
				$(SRCdir)/sparse_matrix.c $(SRCdir)/report_results.c \
				$(SRCdir)/mytimer.c  $(SRCdir)/utils.c \
				$(SRCdir)/sorting_utils.c \
				#$(SRCdir)/hw_counters_ve.c

COMMON_OBJ = $(COMMON_SRC:$(SRCdir)/%.c=bin/%.o)

KERNELS_SRC = $(SRCdir)/bfs_par_kernels.c
KERNELS_OBJ = $(KERNELS_SRC:$(SRCdir)/%.c=bin/%.o)


all: $(TARGETS)
PHONY: objclean clean

# target: pattern: prereq-pattern
$(COMMON_OBJ): bin/%.o: $(SRCdir)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(KERNELS_OBJ): bin/%.o: $(SRCdir)/%.c
	$(LLVM) $(LLVM_FLAGS) -c $< -o $@

%_ver: objclean $(COMMON_OBJ) $(SRCdir)/bfs_%.c
	$(CC) $(CFLAGS) -c $(SRCdir)/bfs_$*.c -o bin/bfs_$*.o
	$(CC) $(CFLAGS) -o bin/bfs_$*$(BIN_SUFFIX) $(COMMON_OBJ) bin/bfs_$*.o $(BFS_LIBS)
	$(CC) $(CFLAGS) -S $(MASM) -o asm/bfs_$*$(BIN_SUFFIX) $(SRCdir)/bfs_$*.c $(BFS_LIBS)

%_par: objclean $(COMMON_OBJ) $(KERNELS_OBJ) $(SRCdir)/bfs_%.c
	$(CC) $(CFLAGS) -c $(SRCdir)/bfs_$*.c -o bin/bfs_$*.o
	$(CC) $(CFLAGS) -o bin/bfs_$*$(BIN_SUFFIX) $(COMMON_OBJ) $(KERNELS_OBJ) bin/bfs_$*.o $(BFS_LIBS)
	$(CC) $(CFLAGS) -S $(MASM) -o asm/bfs_$*$(BIN_SUFFIX) $(SRCdir)/bfs_$*.c $(BFS_LIBS)

%_rv: objclean $(COMMON_OBJ) $(SRCdir)/bfs_%.c
	$(CC) $(CFLAGS) -c $(SRCdir)/bfs_$*.c -o bin/bfs_$*.o
	$(CC) $(CFLAGS) -o bin/bfs_$*$(BIN_SUFFIX) $(COMMON_OBJ) $(KERNELS_RV) bin/bfs_$*.o $(BFS_LIBS)

objclean:
	rm -f bin/*.o

clean: 
	rm bin/bfs_* asm/bfs_* rm bin/*.o
