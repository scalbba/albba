#include "common.h"
#include "utils.h"
#include "macros.h"

int validate_vector(const elem_t *y, const elem_t *y_ref, const int size)
{
    /*
        Input two vectors that are supposed to have the same content.
        Useful to compare floating point accuracy deviation.
        If the different 
    */
    int debug = 1;
    int is_100_correct = 1, is_good_enough = 1, correct = 0;
    elem_t abs_diff, rel_diff;
    elem_t threshold = 1e-7;
    int max_errors = 50;
    int err_count = 0;

    int v = 0;
    while (v < size)
    {
        correct = y_ref[v] == y[v];

        if (!correct || (y[v] != y[v])) // if y is NaN y != y will be true
        {
            is_100_correct = 0;

            abs_diff = y_ref[v] - y[v];
            rel_diff = abs_diff / y_ref[v];
            if (fabs(rel_diff) > threshold || (y[v] != y[v]))
            {
                is_good_enough = 0;
                fprintf(stderr, "Warning: element in y[%d] has a relative diff = <%f>!\n", v, rel_diff);
                if (debug)
                    fprintf(stderr, "   yref: <%g> != <%g>\n", y_ref[v], y[v]);

                err_count++;
                if (err_count > max_errors)
                    return 0;
            }
        }

        v++;
    }

    // Return 2 if 100% correct, 1 if good enough, 0 if not correct
    return is_100_correct + is_good_enough;
}

void init_x(elem_t *x, int n, int test_case)
{
    int i;
    switch (test_case)
    {
    case 1:
        for (i = 0; i < n; i++)
            x[i] = 3.0;
        break;
    case 2:
        for (i = 0; i < n; i++)
            x[i] = (elem_t)i + 1;
        // x[i] = (elem_t) 1.0;
        break;
    default:
        printf("Unexpected X Initialization\n");
        exit(-1);
    }
}

void memset_float(elem_t *vec, float value, int n)
{
    for (int i = 0; i < n; i++)
        vec[i] = value;
}

void print_arr_uint(int N, char *name, int *vector)
{
    int i;
    fprintf(stderr, "\nPrinting vector: %s\n", name);
    for (i = 0; i < N; i++)
    {
        if (!i % 30)
            fprintf(stderr, "\n");
        fprintf(stderr, "%lu, ", vector[i]);
    }
    fprintf(stderr, "\n");
}

void print_arr_float(int N, char *name, elem_t *vector)
{
    int i;
    printf("\nPrinting vector: %s", name);
    for (i = 0; i < N; i++)
    {
        if (!i % 30)
            printf("\n");
        printf("%g, ", vector[i]);
    }
    printf("\n");
}

void check_mem_alloc(void *ptr, const char *err_msg)
{
    if (ptr == NULL)
    {
        fprintf(stderr, "Memory Allocation Error: could not allocate %s. Application will exit.", err_msg);
        exit(1);
    }
}

int *get_rows_size(const int *__restrict row_ptrs, const int nrows, int * padded_size)
{
    int *rs = (int *)malloc((nrows * sizeof(int)));
    check_mem_alloc(rs, "rows_size\n");

    // int vsize = get_multiple_of_align_size(((nrows + 255) / 256) * 256 * sizeof(int));
    // assert((vsize % 2048) == 0);
    for (int i = 0; i < nrows; i++)
    {
        rs[i] = row_ptrs[i + 1] - row_ptrs[i];
    }

    return rs;
}

int *get_rows_size_sorted(const int *row_order,
                             const int *__restrict row_ptrs, 
                             const int nrows, 
                             int * padded_size)
{
    int *rs = (int *)malloc(nrows * sizeof(int));
    check_mem_alloc(rs, "rows_size\n");
    for (int i = 0; i < nrows; i ++)
    {
        int pos = row_order[i];
        rs[i] = row_ptrs[pos + 1] - row_ptrs[pos];
    }
    return rs;
}

int **get_rows_size_perblock(const SparseMatrixCSR *csr_matrix, const uint64_t num_blocks)
{
    int block_size = (csr_matrix->ncolumns + num_blocks - 1) / num_blocks;
#ifdef SPMV_DEBUG
    fprintf(stderr, "The X vector block width is: %lu elements\n", block_size);
#endif

    int vsize = ((csr_matrix->nrows + 255) / 256) * 256 * sizeof(int);
    assert((vsize % 2048) == 0);

    int *rs = (int *)malloc((num_blocks * vsize));
    memset(rs, 0, num_blocks * vsize); // To take into account padded 0 final slice rows.

    int **row_sizes = (int **)malloc((num_blocks * sizeof(int *)));
    check_mem_alloc(row_sizes, "row_sizes\n");

    // Init Array of Arrays pointers
    for (int b = 0; b < num_blocks; b++)
    {
        row_sizes[b] = &rs[b * csr_matrix->nrows];
        memset(row_sizes[b], 0, vsize);
    }
    // This could be improved with some kind of binary search?
    for (int i = 0; i < csr_matrix->nrows; i++)
    {
        /* For each row, find the block where the element belongs in the new blocked matrix and add it to the size count. */
        for (int j = csr_matrix->row_pointers[i]; j < csr_matrix->row_pointers[i + 1]; j++)
        {
            int col_idx = csr_matrix->column_indices[j];
            int block_idx = col_idx / block_size;
            row_sizes[block_idx][i]++;
        }
        // fprintf(stderr, "Row [%" PRIu64 "] has size [%" PRIu64 "]\n", i, sizes[i]);
    }

    return row_sizes;
}

int get_num_verticalops(const int *__restrict vrows_size, const int nrows, const int vlen, int *__restrict slice_width)
{
    // Pre: rows_size is ordered in descent in blocks of vlen size. Meaning every multiple of 256 in rows_size,
    //      including 0, contains the max row size of that block of rows.
    //      e.g: rows_size[0] contains the highest row size between rows_size[0] to rows_size[0+vlen-1];

    // Post:  Returns the total number of vertical ops required. We need this to allocate the vactive_lanes.

    int total_vops = 0;
    int vb_idx = 0;
    for (int i = 0; i < nrows; i += vlen)
    {
        total_vops += vrows_size[i];
        slice_width[vb_idx++] = vrows_size[i];
    }

    return total_vops;
}

void set_active_lanes(const int *__restrict vrows_size, const int nrows,
                      const uint32_t vlen, uint8_t *__restrict vactive_lanes)
{
    int vactive_idx = 0;

    for (int64_t i = 0; i < nrows; i += vlen)
    {
        // last_row = min(nrows, i+vlen) - 1;
        int last_row = ((i + vlen) > nrows) ? (nrows - (uint64_t)1) : (i + vlen - 1);

        // prev_rsize = 0 causes to write first as many vactive_lanes as the minimum row size (which is at vrow_size[end]).
        int prev_rsize = 0;
        int lanes_to_disable = 0;

        // current_lanes = min(vlen, end - i); NOTE: You must use current lanes + 1 on the solver.
        uint8_t current_lanes = (uint8_t)(last_row - i);

        // For each row in this slice:
        for (int64_t j = last_row; j >= i; j--)
        {
            // Traverse backwards
            int current_rsize = vrows_size[j];
            int diff = current_rsize - prev_rsize;

            if (diff > 0)
            {
                current_lanes -= lanes_to_disable;
                DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 "): Adding [%" PRIu64 "-%" PRIu64 "] vops using [%" PRIu8 "] lanes\n", j, current_rsize, vactive_idx, vactive_idx + diff, current_lanes + 1));
                for (int k = 0; k < diff; k++)
                {
                    vactive_lanes[vactive_idx++] = current_lanes;
                }

                lanes_to_disable = 1;
                prev_rsize = current_rsize;
            }
            else
            {
                DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 ") wont add anything.\n", j, current_rsize));
                lanes_to_disable++;
            }
        }
    }
}

void set_slice_vop_length(const int *__restrict rows_size, const int slice_height, uint8_t *__restrict vop_lengths)
{
    int vactive_idx = 0;

    int last_row = slice_height - 1;
    int prev_rsize = 0;
    int lanes_to_disable = 0;

    uint8_t current_lanes = slice_height - 1; // NOTE: You must use current lanes + 1 on the solver.

    for (int64_t j = last_row; j >= 0; j--)
    {
        // Traverse backwards
        int rsize = rows_size[j];
        int diff = rsize - prev_rsize;

        if (diff > 0)
        {
            current_lanes -= lanes_to_disable;
            DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 "): Adding [%" PRIu64 "-%" PRIu64 "] vops using [%" PRIu8 "] lanes\n", j, current_rsize, vactive_idx, vactive_idx + diff, current_lanes + 1));
            // Insert VOPS with the same vop lengths
            for (int k = 0; k < diff; k++)
            {
                vop_lengths[vactive_idx++] = current_lanes;
            }

            lanes_to_disable = 1;
            prev_rsize = rsize;
        }
        else
        {
            DEBUG_MSG(fprintf(stderr, "Row[%" PRIu64 "](%" PRIu64 ") wont add anything.\n", j, current_rsize));
            lanes_to_disable++;
        }
    }
}

bool spmspv_check(bool *mask, 
                  int *colidx_y, elem_t *y, int *nnzy, 
                  int *colidx_x, elem_t *x, int *nnzx)
{
    bool flag = 0;
    (*nnzx) = 0;
    for (int i = 0; i < (*nnzy); i ++)
    {
        const int key = colidx_y[i];
        const int pos = key / 64;
        // unsigned long int bit_mask =  1UL << (63 - (key % 64));
        // if ((mask[pos] & bit_mask) >> (63 - (key % 64)) == 0)
        if (mask[key] == 0)
        {
            colidx_x[(*nnzx)] = key;
            x[(*nnzx)] = y[i];
            (*nnzx) ++;
            flag = 1;
            // mask[pos] |= (1UL << (63 - (key % 64)));
            mask[key] = 1;
        }
    }
    return flag;
}

bool spmv_check(bool * mask,
                elem_t *y,
                elem_t *x,
                int nrows,
                int *nnzx)
{
    bool flag = 0;
    int nnz_tmp = 0;
    for (int i = 0; i < nrows; i ++)
    {
        x[i] =y[i];
        int pos = i / 64;
        // unsigned long int bit_mask =  1UL << (63 - (i % 64));
        // if((mask[pos] & bit_mask) >> (63 - (i % 64)) == 0 && y[i] != 0)
        if (mask[i] == 0 && y[i] != 0)
        {
            mask[i] = 1;
            // mask[pos] |= (1UL << (63 - (i % 64)));;
            flag = 1;
            nnz_tmp ++;
        }
        else if (mask[i] != 0 && y[i] != 0)
        // else if((mask[pos] & bit_mask) >> (63 - (i % 64)) != 0  && y[i] != 0)
        {
            x[i] = 0;
        }
    }
    (*nnzx) = nnz_tmp;
    return flag;
}

void exclusive_scan(int *input, int length)
{
    if (length == 0 || length == 1)
        return;
    int old_value, new_value;
    old_value = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i ++)
    {
        new_value = input[i];
        input[i] = old_value + input[i - 1];
        old_value = new_value;
    }
}

int MergeArr(int* a, int alen, int* b, int blen, int* c)
{
	int i = 0;
	int j = 0;
	int k = 0;
	
	while (i != alen && j != blen)
	{
		if (a[i] < b[j])
			c[k++] = a[i++];
		else if(a[i] == b[j])
		{
		    c[k++] = b[j++];
		    i++;
		}
		else
			c[k++] = b[j++];	
	}
	if (i == alen)
	{
		while (j != blen)
		c[k++] = b[j++];
	}
	else
	{
		while (i != alen)
			c[k++] = a[i++];
	}
	return k;
}
void init_bfs(elem_t *bfs_x, int *colidx_x, int *nnzx, 
              int *num_iter, int *num_vex, int *frontier, 
              double *bfs_time,
              int *iter_times,
              double *conv_time,
              int nrows)
{
    memset(bfs_x, 0, nrows * sizeof(elem_t));
    memset(colidx_x, 0, nrows * sizeof(int));
    (*nnzx) = 1;
    bfs_x[0] = 1;
    colidx_x[0] = 0;
    (*num_iter) = 0;
    (*num_vex) = 1;
    memset(frontier, 0, nrows * sizeof(int));
    (*bfs_time) =0;
    memset(iter_times, 0, nrows * sizeof(double));
    (*conv_time)= 0;
}

void merge_arr(int *arr_a, int *arr_b, int *res,
               int size_a, int size_b, int *res_size,
               size_t *mask, int *merge_row_order, int vlen)
{
    int i = 0, j = 0, k = 0;
    int a, b;
    // int pos;
    // int slice_in_mask;
    // int inner_in_mask;


    while(i < size_a && j < size_b)
    {
        a = arr_a[i];
        b = arr_b[j];
        if (a < b)
        {
            res[k++] = arr_a[i++];
        }
        else if (a > b){
            res[k++] = arr_b[j++];
            // pos = merge_row_order[b];
            // slice_in_mask = pos / vlen;
            // inner_in_mask = pos % vlen;
            // int mask_pos = slice_in_mask * 4 + inner_in_mask / 64;
            // mask[mask_pos] |=  (1UL << (63 - (inner_in_mask % 64)));

        }
        else{
            res[k++] = arr_a[i++];
            j ++;
        }
    }
    while (i < size_a)
    {
        res[k++] = arr_a[i++];
    }

    while (j < size_b)
    {
        b = arr_b[j];
        res[k++] = b;
        // pos = merge_row_order[b];
        // slice_in_mask = pos / vlen;
        // inner_in_mask = pos % vlen;
        // int mask_pos = slice_in_mask * 4 + inner_in_mask / 64;
        // mask[mask_pos] |=  (1UL << (63 - (inner_in_mask % 64)));

        j ++;
    }
    (*res_size) = k;
}
