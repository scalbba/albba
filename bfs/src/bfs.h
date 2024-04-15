
#ifndef _SPMV_H_
#define _SPMV_H_
#include "sparse_matrix.h"
#include "macros.h"

extern size_t align_size;
extern int perf_analysis;
extern int verify;



void run_custom_bfs_test(const SparseMatrixCSR * matrix, 
                         elem_t *bfs_x, 
                         int *colidx_x, 
                         int *nnzx, 
                         int *frontier,
                         int *num_vex,
                         double *conv_time,
                         double *iter_times, 
                         double *bfs_time, 
                         int *num_iter,
                         bool bit_flag,
                         bool sort_flag);

void print_additional_custom_report(char * text_padding, double *elapsed_times);


#endif
