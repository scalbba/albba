#ifndef SPARSE_MATRIX_C
#define SPARSE_MATRIX_C

#include "common.h"
#include "sparse_matrix.h"
#include "macros.h"
// #include "mmio.h"
#include "mytimer.h"
#include "utils.h"

void order_coo_by_column(const SparseMatrixCOO *coo_matrix, SparseMatrixCOO *ordered_coo_matrix)
{
    int *histogram_columns = (int *)aligned_alloc (align_size, coo_matrix->ncolumns * sizeof(int));

    ordered_coo_matrix->nnz = coo_matrix->nnz;
    ordered_coo_matrix->nrows = coo_matrix->nrows;
    ordered_coo_matrix->ncolumns = coo_matrix->ncolumns;

    for (int i = 0; i < coo_matrix->nnz; i++)
    {
        histogram_columns[coo_matrix->columns[i]]++;
    }

    int accum = 0;
    int tmp;
    for (int i = 0; i < coo_matrix->ncolumns; i++)
    {
        tmp = histogram_columns[i];
        histogram_columns[i] = accum;
        accum += tmp;
    }

    for (int i = 0; i < coo_matrix->nnz; i++)
    {
        int insert_index = histogram_columns[coo_matrix->columns[i]];

        ordered_coo_matrix->values[insert_index] = coo_matrix->values[i];
        ordered_coo_matrix->columns[insert_index] = coo_matrix->columns[i];
        ordered_coo_matrix->rows[insert_index] = coo_matrix->rows[i];

        histogram_columns[coo_matrix->columns[i]]++;
    }

    free(coo_matrix->rows);
    free(coo_matrix->columns);
    free(coo_matrix->values);
}

int check_column_order(const SparseMatrixCOO *matrix)
{
    int is_ordered = 1;
    for (int i = 1; i < matrix->nnz; i++)
    {
        if (matrix->columns[i] < matrix->columns[i - 1])
        {
            // fprintf(stderr, "WARNING: Matrix is NOT ordered @ element: %lu, row: %lu, col: %lu\n", i, matrix->rows[i], matrix->columns[i]);
            is_ordered = 0;
            break;
        }
    }

    return is_ordered;
}

void convert_coo_to_csc(const SparseMatrixCOO *coo_matrix, SparseMatrixCSC *csc_matrix)
{
    int col_nnz, cum_sum;

#ifdef DEBUG
    double start, elapsed;
    start = mytimer();
#endif

    /* Allocate CSR Matrix data structure */

    csc_matrix->column_pointers = (int *)aligned_alloc(align_size, (coo_matrix->ncolumns + 1) * sizeof(int));
    check_mem_alloc(csc_matrix->column_pointers, "SparseMatrixCSC.column_pointers");
    memset(csc_matrix->column_pointers, 0, (coo_matrix->ncolumns + 1) * sizeof(int));

    csc_matrix->row_indices = (int *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(int));
    check_mem_alloc(csc_matrix->row_indices, "SparseMatrixCSC.row_indices");

    csc_matrix->values = (elem_t *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(elem_t));
    check_mem_alloc(csc_matrix->values, "SparseMatrixCSC.values");

    csc_matrix->name = coo_matrix->name;
    csc_matrix->nrows = coo_matrix->nrows;
    csc_matrix->ncolumns = coo_matrix->ncolumns;
    csc_matrix->nnz = coo_matrix->nnz;

    // Store the number of Non-Zero elements in each column (histogram)
    for (int i = 0; i < coo_matrix->nnz; i++)
        csc_matrix->column_pointers[coo_matrix->columns[i]]++;

    /*  
        Pre: Array containing the number of nnz in each column.
        Post: csc_matrix->column_pointers constains the (nnz) index to the first element of each column.
    */
    cum_sum = 0;
    for (int i = 0; i < coo_matrix->ncolumns; i++)
    {
        col_nnz = csc_matrix->column_pointers[i];
        csc_matrix->column_pointers[i] = cum_sum;
        cum_sum += col_nnz;
    }
    csc_matrix->column_pointers[csc_matrix->ncolumns] = csc_matrix->nnz;

    int *tmp_col_index = (int *)calloc(coo_matrix->ncolumns, sizeof(int));
    check_mem_alloc(tmp_col_index, "tmp_col_index");

    for (int c = 0; c < csc_matrix->ncolumns; c++)
        tmp_col_index[c] = csc_matrix->column_pointers[c];

    for (int i = 0; i < coo_matrix->nnz; i++)
    {
        int nnz_idx = tmp_col_index[coo_matrix->columns[i]]++;

        csc_matrix->row_indices[nnz_idx] = coo_matrix->rows[i];
        csc_matrix->values[nnz_idx] = coo_matrix->values[i];
    }

    free(tmp_col_index);

#ifdef DEBUG
    elapsed = mytimer() - start;
    fprintf(stderr, "Elapsed time converting COO to CSR =\t %g milliseconds\n", elapsed);
#endif
}

void convert_coo_to_csr(const SparseMatrixCOO *coo_matrix, SparseMatrixCSR *csr_matrix, int free_coo)
{

#ifdef SPMV_DEBUG
    double start, elapsed;
    start = mytimer();
#endif

    /* Allocate CSR Matrix data structure in memory */
    csr_matrix->row_pointers = (int *)aligned_alloc(align_size, (coo_matrix->nrows + 1) * sizeof(int));
    check_mem_alloc(csr_matrix->row_pointers, "SparseMatrixCSR.row_pointers");
    memset(csr_matrix->row_pointers, 0, (coo_matrix->nrows + 1) * sizeof(int));

    csr_matrix->column_indices = (int *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(int));
    check_mem_alloc(csr_matrix->column_indices, "SparseMatrixCSR.column_indices");

    csr_matrix->values = (elem_t *)aligned_alloc(align_size, coo_matrix->nnz * sizeof(elem_t));
    check_mem_alloc(csr_matrix->values, "SparseMatrixCSR.values");

    // Store the number of Non-Zero elements in each Row
    for (int i = 0; i < coo_matrix->nnz; i++)
        csr_matrix->row_pointers[coo_matrix->rows[i]]++;

    // Update Row Pointers so they consider the previous pointer offset
    // (using accumulative sum).
    int cum_sum = 0;
    for (int i = 0; i < coo_matrix->nrows; i++)
    {
        int row_nnz = csr_matrix->row_pointers[i];
        csr_matrix->row_pointers[i] = cum_sum;
        cum_sum += row_nnz;
    }

    /*  Adds COO values to CSR

        Note: Next block of code reuses csr->row_pointers[] to keep track of the values added from
        the COO matrix.
        This way is able to create the CSR matrix even if the COO matrix is not ordered by row.
        In the process, it 'trashes' the row pointers by shifting them one position up.
        At the end, each csr->row_pointers[i+1] should be in csr->row_pointers[i] */

    for (int i = 0; i < coo_matrix->nnz; i++)
    {
        int row_index = coo_matrix->rows[i];
        int column_index = coo_matrix->columns[i];
        elem_t value = coo_matrix->values[i];

        int j = csr_matrix->row_pointers[row_index];
        csr_matrix->column_indices[j] = column_index;
        csr_matrix->values[j] = value;
        csr_matrix->row_pointers[row_index]++;
    }

    // Restore the correct row_pointers
    for (int i = coo_matrix->nrows - 1; i > 0; i--)
    {
        csr_matrix->row_pointers[i] = csr_matrix->row_pointers[i - 1];
    }
    csr_matrix->row_pointers[0] = 0;
    csr_matrix->row_pointers[coo_matrix->nrows] = coo_matrix->nnz;

    csr_matrix->nnz = coo_matrix->nnz;
    csr_matrix->nrows = coo_matrix->nrows;
    csr_matrix->ncolumns = coo_matrix->ncolumns;
    csr_matrix->name = coo_matrix->name;

    /*  For each row, sort the corresponding arrasy csr.column_indices and csr.values

        TODO: We should check if this step makes sense or can be optimized
        1) If the .mtx format by definition is ordered 
        2) If we force the COO Matrix to be ordered first, we can avoid this
        3) Test speed of standard library sorting vs current sorting approach. */

    // for (int i = 0; i < csr_matrix->nrows; i++)
    // {
    //     // print_arr_uint(csr_matrix->row_pointers[i+1]- csr_matrix->row_pointers[i], "before", &csr_matrix->column_indices[csr_matrix->row_pointers[i]]);

    //     // This could be optimized
    //     // sort_paired_vectors(csr_matrix->row_pointers[i], csr_matrix->row_pointers[i + 1],
    //     //                     csr_matrix->column_indices, csr_matrix->values);

    //     // fprintf(stderr, "Sorting from: [%" PRIu64 "] to [%" PRIu64 "]\n", csr_matrix->row_pointers[i],  csr_matrix->row_pointers[i + 1]);

    //     // radix_sort_paired_vectors(csr_matrix->column_indices, csr_matrix->values, csr_matrix->row_pointers[i], csr_matrix->row_pointers[i + 1]);
    //     // print_arr_uint(csr_matrix->row_pointers[i+1]- csr_matrix->row_pointers[i], "after", &csr_matrix->column_indices[csr_matrix->row_pointers[i]]);
    //     // exit(0);
    // }

#ifdef SPMV_DEBUG
    elapsed = mytimer() - start;
    fprintf(stderr, "Elapsed time converting COO to CSR =\t %g seconds\n", elapsed);
#endif

    if (free_coo)
    {
        free(coo_matrix->values);
        free(coo_matrix->rows);
        free(coo_matrix->columns);
    }
}
void convert_csr_to_csc(const SparseMatrixCSR *csr_matrix, SparseMatrixCSC *csc_matrix)
{
#ifdef SPMV_DEBUG
    double start, elapsed;
    start = mytimer();
#endif
     /* Allocate CSC Matrix data structure */
    csc_matrix->column_pointers = (int *)aligned_alloc(align_size, (csr_matrix->ncolumns + 1) * sizeof(int));
    check_mem_alloc(csc_matrix->column_pointers, "SparseMatrixCSC.column_pointers");
    memset(csc_matrix->column_pointers, 0, (csr_matrix->ncolumns + 1) * sizeof(int));

    csc_matrix->row_indices = (int *)aligned_alloc(align_size, csr_matrix->nnz * sizeof(int));
    check_mem_alloc(csc_matrix->row_indices, "SparseMatrixCSC.row_indices");

    csc_matrix->values = (elem_t *)aligned_alloc(align_size, csr_matrix->nnz * sizeof(elem_t));
    check_mem_alloc(csc_matrix->values, "SparseMatrixCSC.values");

    csc_matrix->name = csr_matrix->name;
    csc_matrix->nrows = csr_matrix->nrows;
    csc_matrix->ncolumns = csr_matrix->ncolumns;
    csc_matrix->nnz = csr_matrix->nnz;

    for (int i = 0; i < csr_matrix->nnz; i++)
    {
        csc_matrix->column_pointers[csr_matrix->column_indices[i]]++;
    }

    //prefix sum of csc column pointers
    int old_val, new_val;
    old_val = csc_matrix->column_pointers[0];
    csc_matrix->column_pointers[0] = 0;
    for (int i = 1; i < csc_matrix->ncolumns + 1; i++)
    {
        new_val = csc_matrix->column_pointers[i];
        csc_matrix->column_pointers[i] = old_val + csc_matrix->column_pointers[i-1];
        old_val = new_val;
    }

    int *cscColIncr = (int *)aligned_alloc(align_size, sizeof(int) * (csr_matrix->ncolumns + 1));
    memcpy (cscColIncr, csc_matrix->column_pointers, sizeof(int) * (csr_matrix->ncolumns + 1));

    // insert nnz to csc
     for (int i = 0; i < csr_matrix->nrows; i ++)
     {
         for (int j = csr_matrix->row_pointers[i]; j < csr_matrix->row_pointers[i + 1]; j ++)
         {
             int col = csr_matrix->column_indices[j];
             csc_matrix->row_indices[cscColIncr[col]] = i;
             csc_matrix->values[cscColIncr[col]] = csr_matrix->values[j];
             cscColIncr[col]++;
         }
     }
     free (cscColIncr);


#ifdef DEBUG
    elapsed = mytimer() - start;
    fprintf(stderr, "Elapsed time converting CSR to CSC =\t %g milliseconds\n", elapsed);
#endif

}

void convert_csr_to_ellpack(const SparseMatrixCSR *csr_matrix, SparseMatrixELLPACK *ellpack_matrix,
                            uint32_t order_by_row_size, const int *__restrict row_order)
{

#ifdef DEBUG
    double start, elapsed;
    start = mytimer();
#endif

    /* Allocate ELLPACK Matrix data structure in memory */
    ellpack_matrix->column_indices = (int *)aligned_alloc(align_size, ellpack_matrix->max_row_size * csr_matrix->nrows * sizeof(int));
    check_mem_alloc(ellpack_matrix->column_indices, "SparseMatrixELLPACK.column_indices");
    memset(ellpack_matrix->column_indices, 0, ellpack_matrix->max_row_size * csr_matrix->nrows * sizeof(int));

    ellpack_matrix->values = (elem_t *)aligned_alloc(align_size, ellpack_matrix->max_row_size * csr_matrix->nrows * sizeof(elem_t));
    check_mem_alloc(ellpack_matrix->values, "SparseMatrixELLPACK.values");
    memset(ellpack_matrix->values, 0, ellpack_matrix->max_row_size * csr_matrix->nrows * sizeof(int));

    if (order_by_row_size)
    {
        /*  Adds CSR values and column indices to ELLPACK */
        for (int r = 0; r < csr_matrix->nrows; r++)
        {
            int ell_elem_idx = r * ellpack_matrix->max_row_size;
            int nnz_start = csr_matrix->row_pointers[row_order[r]];
            int nnz_end = csr_matrix->row_pointers[row_order[r] + 1];

            for (int i = nnz_start; i < nnz_end; i++)
            {
                ellpack_matrix->values[ell_elem_idx] = csr_matrix->values[i];
                ellpack_matrix->column_indices[ell_elem_idx] = csr_matrix->column_indices[i];

                ell_elem_idx++;
            }
        }
    }
    else
    {
        /*  Adds CSR values and column indices to ELLPACK */
        for (int r = 0; r < csr_matrix->nrows; r++)
        {
            int ell_elem_idx = r * ellpack_matrix->max_row_size;
            for (int i = csr_matrix->row_pointers[r]; i < csr_matrix->row_pointers[r + 1]; i++)
            {
                ellpack_matrix->values[ell_elem_idx] = csr_matrix->values[i];
                ellpack_matrix->column_indices[ell_elem_idx] = csr_matrix->column_indices[i];

                ell_elem_idx++;
            }
        }
    }

    ellpack_matrix->nnz = csr_matrix->nnz;
    ellpack_matrix->nrows = csr_matrix->nrows;
    ellpack_matrix->ncolumns = csr_matrix->ncolumns;
    ellpack_matrix->name = csr_matrix->name;

    /*  FREE CSR STRUCTURES  */
    free(csr_matrix->values);
    free(csr_matrix->column_indices);
    free(csr_matrix->row_pointers);
}

void convert_csr_to_ell_c_sigma(const SparseMatrixCSR *csr_matrix, SparseMatrixELLPACK *ellpack_matrix,
                                const int *__restrict row_size, const int *__restrict row_order)
{

#ifdef DEBUG
    double start, elapsed;
    start = mytimer();
#endif

    size_t size_of_matrix = 0;
    for (int i = 0; i < csr_matrix->nrows; i += 256)
    {
        size_of_matrix += row_size[i] * 256;
    }
    int size_of_mvalues = size_of_matrix * sizeof(elem_t);
    int size_of_mcolidx = size_of_matrix * sizeof(int);

    // printf("Size allocated: [%" PRIu64 "] of [%" PRIu64 "] minimum needed (nnz:[%" PRIu64 "]). \n", size_of_mvalues, csr_matrix->nnz * sizeof(elem_t),  csr_matrix->nnz); // INSUFICIENT! check why

    /* Allocate ELLPACK Matrix data structure in memory */
    ellpack_matrix->column_indices = (int *)aligned_alloc(align_size, (size_of_mcolidx));
    check_mem_alloc(ellpack_matrix->column_indices, "SparseMatrixELLPACK.column_indices");
    memset(ellpack_matrix->column_indices, 0, size_of_mcolidx);

    ellpack_matrix->values = (elem_t *)aligned_alloc(align_size, (size_of_mvalues)); // as long as elem_t == int
    check_mem_alloc(ellpack_matrix->values, "SparseMatrixELLPACK.values");
    memset(ellpack_matrix->values, 0, size_of_mvalues);

    /* establish row_pointers */
    int *ell_row_ptrs = (int *)aligned_alloc(align_size, ((csr_matrix->nrows) * sizeof(int)));
    check_mem_alloc(ell_row_ptrs, "SparseMatrixELLPACK.ell_row_ptrs");

    int row_offset = 0;
    int block_offset = 0;
    for (int i = 0; i < csr_matrix->nrows; i++)
    {
        int row_block = i / 256;
        int block_size = row_size[row_block * 256];

        row_offset = i % 256;
        ell_row_ptrs[i] = block_offset + row_offset;
        // fprintf(stderr, "element [%" PRIu64 "] @ row [%" PRIu64 "] @ row_block [%" PRIu64 "] @ row size  [%" PRIu64 "]  \n", rpt, i, row_block, block_size);

        if (i % 256 == 255)
        {
            block_offset = block_offset + block_size * 256;
        }
    }

    /*  Adds CSR values and column indices to ELLPACK */

    // print_arr_uint(csr_matrix->nrows, "ellrow", ell_row_ptrs);

    for (int r = 0; r < csr_matrix->nrows; r++)
    {
        int ell_elem_idx = ell_row_ptrs[r];
        int nnz_start = csr_matrix->row_pointers[row_order[r]];
        int nnz_end = csr_matrix->row_pointers[row_order[r] + 1];

        // fprintf(stderr, "start [%" PRIu64 "] to end [%" PRIu64 "] @ row [%" PRIu64 "] @ row order  [%" PRIu64 "]  \n", nnz_start, nnz_end, r, row_order_by_block[r]);
        for (int i = nnz_start; i < nnz_end; i++)
        {
            ellpack_matrix->values[ell_elem_idx] = csr_matrix->values[i];
            ellpack_matrix->column_indices[ell_elem_idx] = csr_matrix->column_indices[i];

            ell_elem_idx += 256;
        }
    }

    ellpack_matrix->nnz = csr_matrix->nnz;
    ellpack_matrix->nrows = csr_matrix->nrows;
    ellpack_matrix->ncolumns = csr_matrix->ncolumns;
    ellpack_matrix->name = csr_matrix->name;

    /*  FREE CSR STRUCTURES  */
    free(csr_matrix->values);
    free(csr_matrix->column_indices);
    free(csr_matrix->row_pointers);
}

void convert_csr_to_sellcs(const SparseMatrixCSR *csr_matrix, SparseMatrixSELLCS *sellcs_matrix,
                           const int *__restrict rows_size, const int *__restrict rows_order, const int freecsr)
{


#ifdef DEBUG
    double start, elapsed;
    start = mytimer();
#endif

    /* Compute slice widths and number of vop (vertical operations) */
    sellcs_matrix->nslices = (csr_matrix->nrows + sellcs_matrix->C - 1) / sellcs_matrix->C;
    sellcs_matrix->slice_widths = (int *)aligned_alloc(align_size, (sellcs_matrix->nslices * sizeof(int)));
    check_mem_alloc(sellcs_matrix->slice_widths, " SparseMatrixSELLCS.slice_widths");
    sellcs_matrix->slice_widths_pointers = (int *)aligned_alloc(align_size, ((sellcs_matrix->nslices + 1) * sizeof(int)));
    check_mem_alloc(sellcs_matrix->slice_widths_pointers, " SparseMatrixSELLCS.slice_widths");

    

    sellcs_matrix->slice_pointers = (int *)aligned_alloc(align_size, ((sellcs_matrix->nslices + 1) * sizeof(int)));
    check_mem_alloc(sellcs_matrix->slice_pointers, "SparseMatrixSELLCS.slice_pointers");

    int size_of_matrix = 0;
    int width_sum = 0;

    for (int s = 0; s < sellcs_matrix->nslices; s++)
    {
        int slice_size = s == sellcs_matrix->nslices - 1 ? csr_matrix->nrows - s * sellcs_matrix->C : sellcs_matrix->C;
        int rsize = rows_size[s * sellcs_matrix->C];
        sellcs_matrix->slice_pointers[s] = size_of_matrix;
        sellcs_matrix->slice_widths[s] = rsize;
        sellcs_matrix->slice_widths_pointers[s] = width_sum;
        width_sum += rsize;
        size_of_matrix += rsize * slice_size;
    }
    sellcs_matrix->slice_pointers[sellcs_matrix->nslices] = size_of_matrix;
    sellcs_matrix->slice_widths_pointers[sellcs_matrix->nslices] = width_sum;
    size_t size_of_mvalues = size_of_matrix * sizeof(elem_t);
    size_t size_of_mcolidx = size_of_matrix * sizeof(int);

    /* Allocate ELLPACK Matrix data structure in memory */
    sellcs_matrix->column_indices = (int *)aligned_alloc(align_size, size_of_matrix * sizeof(int));
    check_mem_alloc(sellcs_matrix->column_indices, "SparseMatrixSELLCS.column_indices");
    memset(sellcs_matrix->column_indices, 0,  size_of_matrix * sizeof(int));

    sellcs_matrix->values = (elem_t *)aligned_alloc(align_size, size_of_matrix * sizeof(elem_t)); // as long as elem_t == int
    check_mem_alloc(sellcs_matrix->values, "SparseMatrixSELLCS.values");
    memset(sellcs_matrix->values, 0, size_of_matrix * sizeof(elem_t));

    int size_of_mask = sellcs_matrix->C / 64 * width_sum;
    // sellcs_matrix->mask = (unsigned long int *)malloc(size_of_mask * sizeof(unsigned long int));
    // memset(sellcs_matrix->mask, 0, size_of_mask * sizeof(unsigned long int));

    /*  Adds CSR values and column indices to SELLCS */
    for (int r = 0; r < csr_matrix->nrows; r++)
    {
        int sidx = r / sellcs_matrix->C;
        int slice_size = (sidx + 1) * sellcs_matrix->C > csr_matrix->nrows ? csr_matrix->nrows - sidx * sellcs_matrix->C : sellcs_matrix->C;
        int ell_elem_idx = sellcs_matrix->slice_pointers[sidx] + (r % sellcs_matrix->C);
        int mask_idx = sellcs_matrix->slice_pointers[sidx] + (r % sellcs_matrix->C);
        int nnz_start = csr_matrix->row_pointers[rows_order[r]];
        int nnz_end = csr_matrix->row_pointers[rows_order[r] + 1];

        // fprintf(stderr, "start [%" PRIu64 "] to end [%" PRIu64 "] @ row [%" PRIu64 "] @ row order  [%" PRIu64 "]  \n", nnz_start, nnz_end, r, row_order_by_block[r]);
        for (int i = nnz_start; i < nnz_end; i++)
        {
            sellcs_matrix->values[ell_elem_idx] = csr_matrix->values[i];
            sellcs_matrix->column_indices[ell_elem_idx] = csr_matrix->column_indices[i];
            int mask_pos = mask_idx / 64;
            // sellcs_matrix->mask[mask_pos] |= 1 << (r % sellcs_matrix->C);
            ell_elem_idx += slice_size;
            mask_idx += sellcs_matrix->C;
        }
    }
    sellcs_matrix->nnz = csr_matrix->nnz;
    sellcs_matrix->nrows = csr_matrix->nrows;
    sellcs_matrix->ncolumns = csr_matrix->ncolumns;
    sellcs_matrix->name = csr_matrix->name;

    /*  FREE CSR STRUCTURES  */
    if (freecsr)
    {
        free(csr_matrix->values);
        free(csr_matrix->column_indices);
        free(csr_matrix->row_pointers);
    }
}

void convert_csr_to_sellcs_dfc(const SparseMatrixCSR *csr_matrix, SparseMatrixSELLCS *sellcs_matrix,
                               const int *__restrict rows_size, const int *__restrict rows_order, const int freecsr)
{
#ifdef DEBUG
    double start, elapsed;
    start = mytimer();
#endif

    /* Compute slice widths and number of vop (vertical operations) */
    sellcs_matrix->nslices = (csr_matrix->nrows + sellcs_matrix->C - 1) / sellcs_matrix->C;
    sellcs_matrix->slice_widths = (int *)aligned_alloc(align_size, sellcs_matrix->nslices * sizeof(int));
    check_mem_alloc(sellcs_matrix->slice_widths, " SparseMatrixSELLCS.slice_widths");
    sellcs_matrix->vop_pointers = (int *)aligned_alloc(align_size, (sellcs_matrix->nslices + 1) * sizeof(int) + 1);
    check_mem_alloc(sellcs_matrix->vop_pointers, " SparseMatrixSELLCS.vop_pointers");
    int slice_idx = 0;
    int vop_count = 0;
    for (int r = 0; r < csr_matrix->nrows; r += sellcs_matrix->C)
    {
        sellcs_matrix->slice_widths[slice_idx] = rows_size[r];
        sellcs_matrix->vop_pointers[slice_idx] = vop_count;
        vop_count += rows_size[r];
        slice_idx++;
    }
    sellcs_matrix->vop_pointers[slice_idx] = vop_count;

    /* Compute the vop lengths and the size of column_indices and values data structures */
    sellcs_matrix->vop_lengths = (uint8_t*)aligned_alloc(align_size, vop_count * sizeof(uint8_t));
    check_mem_alloc(sellcs_matrix->vop_lengths, "SparseMatrixSELLCS.vop_lengths");
    set_active_lanes(rows_size, csr_matrix->nrows, sellcs_matrix->C, sellcs_matrix->vop_lengths);

    sellcs_matrix->slice_pointers = (int *)aligned_alloc(align_size, (sellcs_matrix->nslices +1) * sizeof(int));
    check_mem_alloc(sellcs_matrix->slice_pointers, "SparseMatrixSELLCS.slice_pointers");

    sellcs_matrix->max_index = (int *)aligned_alloc(align_size, vop_count * sizeof(int));
    memset(sellcs_matrix->max_index, 0, vop_count * sizeof(int));
    sellcs_matrix->min_index = (int *)aligned_alloc(align_size, vop_count * sizeof(int));
    memset(sellcs_matrix->min_index, 0, vop_count * sizeof(int));

    int vop_idx = 0;
    int size_of_matrix = 0;

    for (int s = 0; s < sellcs_matrix->nslices; s++)
    {
        sellcs_matrix->slice_pointers[s] = size_of_matrix;
        for (int v = 0; v < sellcs_matrix->slice_widths[s]; v++)
            size_of_matrix += sellcs_matrix->vop_lengths[vop_idx++] + 1;
    }
    sellcs_matrix->slice_pointers[sellcs_matrix->nslices] = size_of_matrix;
    size_t size_of_values = size_of_matrix * sizeof(elem_t);
    size_t size_of_colidx = size_of_matrix * sizeof(int);

    /* Allocate SELLCS NNZ data structures in memory */
    sellcs_matrix->column_indices = (int *)aligned_alloc(align_size, size_of_colidx);
    check_mem_alloc(sellcs_matrix->column_indices, "SparseMatrixSELLCS.column_indices");

    sellcs_matrix->values = (elem_t *)aligned_alloc(align_size, (size_of_values));
    check_mem_alloc(sellcs_matrix->values, "SparseMatrixSELLCS.values"); 

    /* Copy values and column indices from CSR to SELLCS */
    // int insert_idx = 0;
    #pragma omp parallel for schedule(dynamic, 32)
    for (int s = 0; s < sellcs_matrix->nslices; s++)
    {
        // int sptr = sellcs_matrix->slice_pointers[slice_idx];
        int swidth = sellcs_matrix->slice_widths[s];
        int base_row = s * sellcs_matrix->C;
        int insert_idx = sellcs_matrix->slice_pointers[s];
        int vop_id = sellcs_matrix->vop_pointers[s];

        for (int vop = 0; vop < swidth; vop++)
        {
            int max = 0;
            int min = csr_matrix->nrows;
            for (int r = 0; r < sellcs_matrix->vop_lengths[vop_id] + 1; r++) // Note:  vop length must be + 1
            {
                int csr_nnz_idx = csr_matrix->row_pointers[rows_order[base_row + r]] + vop;
                sellcs_matrix->values[insert_idx] = csr_matrix->values[csr_nnz_idx];
                sellcs_matrix->column_indices[insert_idx] = csr_matrix->column_indices[csr_nnz_idx];
                max = max < sellcs_matrix->column_indices[insert_idx] ? sellcs_matrix->column_indices[insert_idx] : max;
                min = min > sellcs_matrix->column_indices[insert_idx] ? sellcs_matrix->column_indices[insert_idx] : min;
                insert_idx++;
            }
            sellcs_matrix->max_index[vop_id] = max;
            sellcs_matrix->min_index[vop_id] = min;
            vop_id++;
        }
    }

    sellcs_matrix->nnz = csr_matrix->nnz;
    sellcs_matrix->nrows = csr_matrix->nrows;
    sellcs_matrix->ncolumns = csr_matrix->ncolumns;
    sellcs_matrix->name = csr_matrix->name;

    /*  FREE CSR STRUCTURES  */
    if (freecsr)
    {
        free(csr_matrix->values);
        free(csr_matrix->column_indices);
        free(csr_matrix->row_pointers);
    }
}


/*  Create a pentadiagonal matrix, representing very roughly a finite
    difference approximation to the Laplacian on a square n x n mesh */

void load_pentadiagonal(int n, SparseMatrixCSR *csr_matrix)
{
    /* Warning: if n > sqrt(2^31), you will get integer overflow */
    if (n > 2 << 30)
    {
        fprintf(stderr, "Error: Matrix num rows (%lu) exceeds the limit (%d).", n, 2 << 30);
        exit(1);
    }

    if (n < 3)
    {
        fprintf(stderr, "Error: Matrix num rows(=%lu) cannot be smaller than 3!\n", n);
        exit(1);
    }

    csr_matrix->nrows = n * n; // n * n is correct
    csr_matrix->ncolumns = csr_matrix->nrows;
    csr_matrix->nnz = csr_matrix->nrows * 5;

    // ALIGNMENT?
    csr_matrix->row_pointers = (int *)aligned_alloc(align_size, (csr_matrix->nrows + 1) * sizeof(int));
    csr_matrix->column_indices = (int *)aligned_alloc(align_size, csr_matrix->nnz * sizeof(int));
    csr_matrix->values = (elem_t *)aligned_alloc(align_size, csr_matrix->nnz * sizeof(elem_t));

    int row_index = 0;
    int nnz_index = 0;
    int i = 0, j = 0;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            csr_matrix->row_pointers[row_index] = nnz_index;

            // Diagonal row_index-n (starts at row_index = n)
            if (i > 0)
            {
                csr_matrix->column_indices[nnz_index] = row_index - n;
                csr_matrix->values[nnz_index] = -1.0;
                nnz_index++;
            }
            // Diagonal row_index-1
            if (j > 0)
            {
                csr_matrix->column_indices[nnz_index] = row_index - 1;
                csr_matrix->values[nnz_index] = -1.0;
                nnz_index++;
            }

            // Diagonal row_index
            csr_matrix->column_indices[nnz_index] = row_index;
            csr_matrix->values[nnz_index] = 4.0;
            nnz_index++;

            // Diagonal row_index+1
            if (j < n - 1)
            {
                csr_matrix->column_indices[nnz_index] = row_index + 1;
                csr_matrix->values[nnz_index] = -1.0;
                nnz_index++;
            }

            // Diagonal row_index+n (ends at row_index = nrows-n)
            if (i < n - 1)
            {
                csr_matrix->column_indices[nnz_index] = row_index + n;
                csr_matrix->values[nnz_index] = -1.0;
                nnz_index++;
            }
            row_index++;
        }
    }
    csr_matrix->row_pointers[row_index] = nnz_index;
    csr_matrix->name = "pentadiagonal";
}

void load_octadiagonal(int n, SparseMatrixCSR *csr_matrix)
{
    /* Warning: if n > sqrt(2^31), you will get integer overflow */
    if (n > 2 << 30)
    {
        fprintf(stderr, "Error: Matrix num rows (%lu) exceeds the limit (%d).", n, 2 << 30);
        exit(1);
    }

    if (n < 3)
    {
        fprintf(stderr, "Error: Matrix num rows(=%lu) cannot be smaller than 3!\n", n);
        exit(1);
    }

    csr_matrix->nrows = n * n; // n * n is correct
    csr_matrix->ncolumns = csr_matrix->nrows;

    csr_matrix->row_pointers = (int *)aligned_alloc(align_size, (csr_matrix->nrows + 1) * sizeof(int));
    csr_matrix->column_indices = (int *)aligned_alloc(align_size, ((csr_matrix->nrows * 8) + 256) * sizeof(int));
    csr_matrix->values = (elem_t *)aligned_alloc(align_size, csr_matrix->nrows * 8 * sizeof(elem_t));

    int row_index = 0;
    int nnz_index = 0;
    int i = 0, j = 0;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            csr_matrix->row_pointers[row_index] = nnz_index;

            // Diagonal row_index-n (starts at row_index = n)
            if (i > 1)
            {
                csr_matrix->column_indices[nnz_index] = row_index - n - n;
                csr_matrix->values[nnz_index] = -4.0000;
                nnz_index++;
            }
            if (i > 0)
            {
                csr_matrix->column_indices[nnz_index] = row_index - n;
                csr_matrix->values[nnz_index] = -3.00000;
                nnz_index++;
            }

            // Diagonal row_index-2
            if (j > 1 || i > 0)
            {
                csr_matrix->column_indices[nnz_index] = row_index - 2;
                csr_matrix->values[nnz_index] = -2.00000;
                nnz_index++;
            }
            // Diagonal row_index-1
            if (j > 0 || i > 0)
            {
                csr_matrix->column_indices[nnz_index] = row_index - 1;
                csr_matrix->values[nnz_index] = -1.00000;
                nnz_index++;
            }

            // Diagonal row_index
            csr_matrix->column_indices[nnz_index] = row_index;
            csr_matrix->values[nnz_index] = 4.0;
            nnz_index++;

            // Diagonal row_index+1
            if ((j < n - 1) && (i < n - 1))
            {
                csr_matrix->column_indices[nnz_index] = row_index + 1;
                csr_matrix->values[nnz_index] = -1.5;
                nnz_index++;
            }

            // Diagonal row_index+n (ends at row_index = nrows-n)
            if (i < n - 1)
            {
                csr_matrix->column_indices[nnz_index] = row_index + n;
                csr_matrix->values[nnz_index] = -2.5;
                nnz_index++;
            }

            // Diagonal row_index+n+n (ends at row_index = nrows-n)
            if (i < n - 2)
            {
                csr_matrix->column_indices[nnz_index] = row_index + n + n;
                csr_matrix->values[nnz_index] = -3.5;
                nnz_index++;
            }

            row_index++;
        }
    }
    csr_matrix->row_pointers[row_index] = nnz_index;
    csr_matrix->name = "octadiagonal";
    csr_matrix->nnz = nnz_index; // This is not ok ???

    for (int k = 0; k < 256; k++)
    {
        csr_matrix->column_indices[nnz_index + k] = 0;
    }
}

void generate_sparse_matrix(int nrows, int ncols, double sparsity, int col_dist, SparseMatrixCSR *__restrict mymatrix)
{
    /*
        col_dist: Determines how columns are distributed
            0: Random
            1: Strided
            2: Strided + Shifted by 1 NOT IMPLEMENTED
            3: Strided + Shifted by Cache line size NOT IMPLEMENTED
    */
    int nnz = 0, stride = 0;
    int nnz_per_row = 0;
    elem_t nnz_per_rowf = 0;

    switch (col_dist)
    {
    case 0:
        /* TODO */

        break;

    case 1:
        nnz_per_rowf = ncols * sparsity;
        stride = ncols / nnz_per_rowf;
        nnz_per_row = ncols / stride;
        nnz = nnz_per_row * nrows;

        if (nnz_per_row < 1)
            exit(1);
        break;

    default:
        break;
    }

    mymatrix->ncolumns = ncols;
    mymatrix->nrows = nrows;
    mymatrix->nnz = nnz;
    mymatrix->name = "strided";

    fprintf(stderr, "Generating matrix with sparsity: %f\n", sparsity);
    fprintf(stderr, "NNZ per row: %lu, w/ stride: %lu, along %lu columns\n", nnz_per_row, stride, ncols);
    fprintf(stderr, "For a total NNZ of: %lu\n", nnz);

    mymatrix->row_pointers = (int *)aligned_alloc(align_size, (mymatrix->nrows + 1) * sizeof(int));
    check_mem_alloc(mymatrix->row_pointers, "SparseMatrixCSR.row_pointers");
    // memset(csr_matrix->row_pointers, 0, (nrows + 1) * sizeof(int));

    mymatrix->column_indices = (int *)aligned_alloc(align_size,  nnz * sizeof(int));
    check_mem_alloc(mymatrix->column_indices, "SparseMatrixCSR.column_indices");

    mymatrix->values = (elem_t *)aligned_alloc(align_size, nnz * sizeof(elem_t));
    check_mem_alloc(mymatrix->values, "SparseMatrixCSR.values");

    int *col_idx_sequence = (int *)aligned_alloc(align_size, nnz_per_row * sizeof(int));
    for (int c = 0; c < nnz_per_row; c++)
    {
        col_idx_sequence[c] = c * stride;
    }
    //  print_arr_uint(nnz_per_row, "index sequence", col_idx_sequence);

    for (int i = 0; i < nnz; i++)
    {
        // fill with random values between 0 and 10
        mymatrix->values[i] = ((elem_t)rand() / (elem_t)(RAND_MAX)*5);
        mymatrix->column_indices[i] = col_idx_sequence[i % nnz_per_row];
    }

    for (int r = 0; r < nrows + 1; r++)
    {
        mymatrix->row_pointers[r] = r * nnz_per_row;
    }

    //print_arr_float(nnz, "values", mymatrix->values);
    // print_arr_uint(nnz, "col indices", mymatrix->column_indices);

    // print_arr_uint(nrows + 1, "rows indices", mymatrix->row_pointers);

    // print_full_csr_matrix(csr_matrix);
}

void print_full_csr_matrix(const SparseMatrixCSR *csr_matrix)
{
    for (int row_index = 0; row_index < csr_matrix->nrows; row_index++)
    {
        int start = csr_matrix->row_pointers[row_index];
        int end = csr_matrix->row_pointers[row_index + 1] - 1;

        int elem_index = start;
        for (int col_index = 0; col_index < csr_matrix->ncolumns; col_index++)
        {

            if (start - end == 0)
            {
                // emtpy row
                printf("0");
                if (col_index < csr_matrix->ncolumns - 1)
                    printf(",");
            }
            else if (col_index < csr_matrix->column_indices[start])
            {
                //leading 0's
                printf("0");
                if (col_index < csr_matrix->ncolumns - 1)
                    printf(",");
            }
            else if (col_index > csr_matrix->column_indices[end])
            {
                //trailing 0's
                printf("0");
                if (col_index < csr_matrix->ncolumns - 1)
                    printf(",");
            }
            else if (csr_matrix->column_indices[elem_index] == col_index)
            {
                //nnz
                printf("%.2e", csr_matrix->values[elem_index]);
                if (col_index < csr_matrix->ncolumns - 1)
                    printf(",");
                elem_index++;
            }
            else
            {
                // 0's between nnz
                printf("0");
                if (col_index < csr_matrix->ncolumns - 1)
                    printf(",");
                //elem_index++;
            }
        }
        printf("\n");
    }
    fflush(stdout);
}


void free_sellcs_struct(SparseMatrixSELLCS* sellcs)
{
    free(sellcs->values);
    free(sellcs->column_indices);
    free(sellcs->slice_pointers);
    free(sellcs->slice_widths);
    free(sellcs->row_order);

#if defined(ENABLE_DFC)
    free(sellcs->vop_lengths);
    free(sellcs->vop_pointers);
#endif
}
#endif

