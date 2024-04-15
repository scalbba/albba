#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "common.h"
extern size_t align_size;
typedef double elem_t;

enum MatrixFormat
{
    COO,
    CSR,
    CSC,
    CSRSIMD,
    ELLPACK,
    VBSF
};

struct SparseMatrixCSR_STRUCT
{
    char *name;
    elem_t *values; // values of matrix entries
    int *column_indices;
    int *row_pointers;
    int nrows;
    int ncolumns;
    int nnz;
    int isSymmetric;
};
typedef struct SparseMatrixCSR_STRUCT SparseMatrixCSR;

struct SparseMatrixELLPACK_STRUCT
{
    char *name;
    elem_t *values;
    int *column_indices;
    int max_row_size;
    int nrows;
    int ncolumns;
    int nnz;
};
typedef struct SparseMatrixELLPACK_STRUCT SparseMatrixELLPACK;

struct SparseMatrixSELLCS_STRUCT
{
    char *name;
    elem_t *values;
    int *column_indices;
    int nrows;
    int ncolumns;
    int nnz;
    int C;
    int sigma;
    int nslices;
    int *slice_widths;
    int *slice_pointers;
    int *row_order;
    uint8_t *vop_lengths;
    int *vop_pointers; // slice widths and vop_pointers can be merged, to express both; just extend vop_pointers size by 1 as row_pointers in csr is nrows + 1;
    int *slice_widths_pointers;   
    int *max_index;
    int *min_index;
};
typedef struct SparseMatrixSELLCS_STRUCT SparseMatrixSELLCS;

struct SparseMatrixSELLCSBLOCK_STRUCT
{
    char *name;
    elem_t *values;
    int *column_indices;
    int nrows;
    int ncolumns;
    int nnz;
    int C;
    int sigma;
    int nblocks;
    int nslices;
    int *slice_widths;
    int *slice_pointers;
    int *block_pointers;
    int *row_order;
    elem_t *y_tmp;
    uint8_t *vop_lengths;
    int *vop_pointers;
    int *slice_block;
};
typedef struct SparseMatrixSELLCSBLOCK_STRUCT SparseMatrixSELLCSBLOCK;

struct SparseMatrixCSC_STRUCT
{
    char *name;
    elem_t *values; // values of matrix entries
    int *row_indices;
    int *column_pointers;
    int nrows;
    int ncolumns;
    int nnz;
};
typedef struct SparseMatrixCSC_STRUCT SparseMatrixCSC;


struct SparseMatrixCOO_STRUCT
{
    char *name;
    elem_t *values;  // values of matrix entries
    int *rows;    // row_index
    int *columns; // col_index
    int nrows;
    int ncolumns;
    int nnz;
};
typedef struct SparseMatrixCOO_STRUCT SparseMatrixCOO;

// void load_from_mtx_file(const char *mtx_filepath, SparseMatrixCOO *coo_matrix);
// void fast_load_from_mtx_file(const char *mtx_filepath, SparseMatrixCOO *coo_matrix);
void order_coo_by_column(const SparseMatrixCOO *coo_matrix, SparseMatrixCOO *ordered_coo_matrix);
void convert_coo_to_csr(const SparseMatrixCOO *coo_matrix, SparseMatrixCSR *csr_matrix, int free_coo);
void convert_coo_to_csc(const SparseMatrixCOO *coo_matrix, SparseMatrixCSC *csc_matrix);
void convert_csr_to_csc(const SparseMatrixCSR *csr_matrix, SparseMatrixCSC *csc_matrix);
void convert_csr_to_ellpack(const SparseMatrixCSR *csr_matrix, SparseMatrixELLPACK *ellpack_matrix, const uint32_t order_by_row_size, const int *__restrict row_order);
void convert_csr_to_ell_c_sigma(const SparseMatrixCSR *csr_matrix, SparseMatrixELLPACK *ellpack_matrix, const int *__restrict__ row_size, const int *__restrict row_order);
void convert_csr_to_blocked_sellcs(const SparseMatrixCSR *csr_matrix, SparseMatrixELLPACK *ellpack_matrix, int **__restrict__ row_sizes, int **__restrict row_order_by_block, const uint64_t num_blocks);

void convert_csr_to_sellcs(const SparseMatrixCSR *csr_matrix, SparseMatrixSELLCS *sellcs_matrix,
                           const int *__restrict rows_size, const int *__restrict rows_order, const int freecsr);

void convert_csr_to_sellcs_dfc(const SparseMatrixCSR *csr_matrix, SparseMatrixSELLCS *sellcs_matrix,
                               const int *__restrict rows_size, const int *__restrict rows_order, const int freecsr);


void load_pentadiagonal(int n, SparseMatrixCSR *csr_matrix);
void load_octadiagonal(int n, SparseMatrixCSR *csr_matrix);
void generate_sparse_matrix(int nrows, int ncols, double sparsity, int col_dist, SparseMatrixCSR *csr_matrix);
void print_full_csr_matrix(const SparseMatrixCSR *csr_matrix);

void free_sellcs_struct(SparseMatrixSELLCS* sellcs);

#endif // SPARSE_MATRIX_H
