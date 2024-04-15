#ifndef BFS_REF_H
#define BFS_REF_H

#include "common.h"
#include "sparse_matrix.h"
#include "utils.h"

extern size_t align_size;
void kernel_csr_spmv(SparseMatrixCSR *csr_matrix,
                     const elem_t *x,
                     elem_t *y)
{
    for (int i = 0; i < csr_matrix->nrows; i++)
    {
        elem_t sum = 0;
        for (int j = csr_matrix->row_pointers[i]; j < csr_matrix->row_pointers[i + 1]; j++)
        {
            int col = csr_matrix->column_indices[j];
            elem_t val = csr_matrix->values[j];
            sum += val * x[col];
        }
        y[i] = sum;
    }
}

void run_bfs_ref(SparseMatrixCSR *csr_matrix,
                 elem_t *x, int *colidx_x, int *nnzx,
                 int *iter_ref,
                 int *sum_vex_ref,
                 int *num_vex_ref)
{
    bool flag = 0;
    elem_t *bfs_y = (elem_t *)aligned_alloc(align_size, csr_matrix->nrows * sizeof(elem_t));
    memset(bfs_y, 0, csr_matrix->nrows * sizeof(elem_t));

    int *colidx_y = (int *)aligned_alloc(align_size, csr_matrix->nrows * sizeof(int));
    memset(colidx_y, 0, csr_matrix->nrows * sizeof(int));

    int nnzy;

    bool *mask_ref = (bool *)aligned_alloc(align_size, (csr_matrix->nrows) * sizeof(bool));
    memset(mask_ref, 0, (csr_matrix->nrows) * sizeof(bool));
    mask_ref[0] = 1;

    struct timeval t1, t2;
    struct timeval t_start, t_end;
    double t_check = 0;

    gettimeofday(&t_start, NULL);
    do
    {
        kernel_csr_spmv(csr_matrix, x, bfs_y);
        num_vex_ref[(*iter_ref)] = (*nnzx);
        gettimeofday(&t1, NULL);
        flag = spmv_check(mask_ref, bfs_y, x, csr_matrix->nrows, nnzx);
        gettimeofday(&t2, NULL);
        t_check += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        memset(bfs_y, 0, csr_matrix->nrows * sizeof(elem_t));
        (*iter_ref)++;
        (*sum_vex_ref) += (*nnzx);

    } while (*nnzx != 0);
    gettimeofday(&t_end, NULL);
    double t_ref = (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    // printf("bfs reference, iter = %i, overall time = %.3f ms, check time = %.3f ms\n", (*iter_ref), t_ref, t_check);
}

#endif