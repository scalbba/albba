#ifndef UTILS_H
#define UTILS_H
#include "common.h"
#include "sparse_matrix.h"

int validate_vector(const elem_t *y, const elem_t *y_ref, const int size);
void init_x(elem_t *x, int n, int test_case);
void memset_float(elem_t *vec, float value, int n);
void print_arr_float(int N, char *name, elem_t *vector);
void print_arr_uint(int N, char *name, int *vector);
void check_mem_alloc(void * ptr, const char * err_msg);


int *get_rows_size(const int *__restrict row_ptrs, const int nrows, int * padded_size);
int *get_rows_size_sorted(const int *row_order, const int *__restrict row_ptrs, const int nrows, int * padded_size);
int **get_rows_size_perblock(const SparseMatrixCSR *csr_matrix, const uint64_t num_blocks);

int get_multiple_of_align_size(int size);
int get_num_verticalops(const int *__restrict vrows_size, const int nrows, const int vlen, int *__restrict vblock_size);
int get_num_verticalops_blocked(int **__restrict vrows_size, const int nrows, const int vlen, int *__restrict vblock_size, const int num_blocks);

void set_active_lanes(const int *__restrict vrows_size, const int nrows, const uint32_t vlen, uint8_t *__restrict vactive_lanes);
void set_slice_vop_length(const int *__restrict rows_size, const int slice_height, uint8_t *__restrict vop_lengths);

bool spmspv_check(bool *mask, int *colidx_y, elem_t *y, int *nnzy, int *colidx_x, elem_t *x, int *nnzx);
bool spmv_check(bool *mask, elem_t *y, elem_t *x, int nrows, int *nnzx);
void exclusive_scan(int *input, int length);
int MergeArr(int* a, int alen, int* b, int blen, int* c);
void init_bfs(elem_t *bfs_x, int *colidx_x, int *nnzx, 
              int *num_iter, int *num_vex, int *frontier, 
              double *bfs_time,
              int *iter_times,
              double *conv_time,             
              int nrows);

void merge_arr(int *arr_a, int *arr_b, int *res,
               int size_a, int size_b, int *res_size,
               size_t *mask, int *merge_row_order, int vlen);

#endif // UTILS_H
