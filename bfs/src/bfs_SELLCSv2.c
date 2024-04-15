#include "common.h"
#include "assert.h"
#include "macros.h"
#include "mytimer.h"
#include "sorting_utils.h"
#include "bfs.h"
#include "bfs_par_kernels.h"
#include "utils.h"

#if defined(USE_OMP)
#include "omp.h"
#if defined(ENABLE_DFC)
#define ALG_VERSION "BFS_DFC_omp"
#else
#define ALG_VERSION "BFS_omp"
#endif
#else
#if defined(ENABLE_DFC)
#define ALG_VERSION "BFS_DFC"
#else
#define ALG_VERSION "BFS"
#endif
#endif

#define USE_OMP
enum MatrixFormat required_matrix_format = CSR;
char algorithm_version[30] = ALG_VERSION;

int sigma_window = 16384;
int chunk_size = 8;
int num_blocks = 1;
extern int bypass_record;

void sellcs_preprocess(const SparseMatrixCSR *restrict csr_matrix, SparseMatrixSELLCS *sellcs_matrix, bool sort_flag)
{
    DEBUG_MSG(fprintf(stderr, "== Start pre-processing [CSR to SELL-C-SIGMA w/ DFC] ==\n"));
    int padded_size;
    int *rows_size;

    if (sort_flag)
    {
        sellcs_matrix->row_order = sort_by_1st_colidx(csr_matrix->row_pointers, csr_matrix->column_indices, csr_matrix->nrows);
        rows_size = get_rows_size_sorted(sellcs_matrix->row_order, csr_matrix->row_pointers, csr_matrix->nrows, &padded_size);
        get_order_by_row_size_radix_sorted(sellcs_matrix->row_order, rows_size, csr_matrix->nrows, sellcs_matrix->sigma);
    }
    else
    {
        rows_size = get_rows_size(csr_matrix->row_pointers, csr_matrix->nrows, &padded_size);
        sellcs_matrix->row_order = get_order_by_row_size_radix(rows_size, csr_matrix->nrows, sellcs_matrix->sigma);
    }

#if defined(ENABLE_DFC)
    convert_csr_to_sellcs_dfc(csr_matrix, sellcs_matrix, rows_size, sellcs_matrix->row_order, 0);
#else
    convert_csr_to_sellcs(csr_matrix, sellcs_matrix, rows_size, sellcs_matrix->row_order, 0);
#endif

    free(rows_size);
}

int *get_task_ptrs(SparseMatrixSELLCS *matrix, int total_tasks)
{
    int total_nnz = matrix->slice_pointers[matrix->nslices];
    int nnz_per_task = ((total_nnz + total_tasks - 1) / total_tasks);
    int *task_ptrs = (int *)aligned_alloc(align_size, (total_tasks + 1) * sizeof(int));

    int task_nnz = 0;
    int task_idx = 0;
    task_ptrs[task_idx++] = 0;

    for (int s = 0; s < matrix->nslices; s++)
    {
        if ((task_nnz >= nnz_per_task))
        {
            // fprintf(stderr, "Task #%lu, starts at slice: %lu\n", task_idx, s);
            task_ptrs[task_idx++] = s;
            task_nnz = 0;
        }
        task_nnz += matrix->slice_pointers[s + 1] - matrix->slice_pointers[s];
    }
    for (int i = task_idx; i < total_tasks + 1; i++)
        task_ptrs[i] = matrix->nslices;

    return task_ptrs;
}
int *get_spmspv_task_ptrs(SparseMatrixSELLCS *matrix, int *slice_nbypass, int slice_num, int total_tasks)
{
    int total_nnz = 0;
    for (int i = 0; i < slice_num; i++)
    {
        int sid = slice_nbypass[i];
        total_nnz += matrix->slice_pointers[sid + 1] - matrix->slice_pointers[sid];
    }
    int nnz_per_task = ((total_nnz + total_tasks - 1) / total_tasks);
    int *task_ptrs = (int *)aligned_alloc(align_size, (total_tasks + 1) * sizeof(int));

    int task_nnz = 0;
    int task_idx = 0;
    task_ptrs[task_idx++] = 0;
    for (int s = 0; s < slice_num; s++)
    {
        int sid = slice_nbypass[s];
        if ((task_nnz >= nnz_per_task))
        {
            task_ptrs[task_idx++] = s;
            task_nnz = 0;
        }
        task_nnz += matrix->slice_pointers[sid + 1] - matrix->slice_pointers[sid];
    }
    task_ptrs[task_idx] = slice_num;
    return task_ptrs;
}

int get_start_index(int *row_order, int nrows)
{
    for (int i = 0; i < nrows; i++)
    {
        if (row_order[i] == 0)
        {
            return i;
        }
    }
    return;
}

void run_custom_bfs_test(const SparseMatrixCSR *csr_matrix,
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
                         bool sort_flag)
{

    struct timeval t1, t2;

    // create mask_x for SpMV

    int *mask_x = (int *)aligned_alloc(align_size, csr_matrix->nrows * sizeof(int));
    memset(mask_x, 0, csr_matrix->nrows * sizeof(int));
    mask_x[0] = 1;
    int *mask_y = (int *)aligned_alloc(align_size, csr_matrix->nrows * sizeof(int));
    memset(mask_y, 0, csr_matrix->nrows * sizeof(int));

    // Allocate bfs_y
    elem_t *bfs_y = (elem_t *)aligned_alloc(align_size, csr_matrix->nrows * sizeof(elem_t));
    memset(bfs_y, 0, csr_matrix->nrows * sizeof(elem_t));
    int *colidx_y = (int *)aligned_alloc(align_size, csr_matrix->nrows * sizeof(int));
    memset(colidx_y, 0, csr_matrix->nrows * sizeof(int));

    int nnzy = 0;
    // Initialize bucket
    int *bucket = (int *)aligned_alloc(align_size, (csr_matrix->nrows + 1) * sizeof(int));
    memset(bucket, 0, (csr_matrix->nrows + 1) * sizeof(int));

    SparseMatrixSELLCS sellcs_matrix;
    sellcs_matrix.C = 256;
    sellcs_matrix.sigma = sigma_window;

    gettimeofday(&t1, NULL);
    sellcs_preprocess(csr_matrix, &sellcs_matrix, sort_flag);
    gettimeofday(&t2, NULL);
    (*conv_time) = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    int *slice_idx_max = (int *)aligned_alloc(align_size, sellcs_matrix.nslices * sizeof(int));
    memset(slice_idx_max, 0, sellcs_matrix.nslices * sizeof(int));
    int *slice_idx_min = (int *)aligned_alloc(align_size, sellcs_matrix.nslices * sizeof(int));
    memset(slice_idx_min, sellcs_matrix.nrows, sellcs_matrix.nslices * sizeof(int));

    for (int i = 0; i < sellcs_matrix.nslices; i++)
    {
        for (int j = sellcs_matrix.vop_pointers[i]; j < sellcs_matrix.vop_pointers[i + 1]; j++)
        {
            slice_idx_max[i] = slice_idx_max[i] < sellcs_matrix.max_index[j] ? sellcs_matrix.max_index[j] : slice_idx_max[i];
            slice_idx_min[i] = slice_idx_min[i] > sellcs_matrix.min_index[j] ? sellcs_matrix.min_index[j] : slice_idx_min[i];
        }
    }
    int mask_size = sellcs_matrix.nrows / 64 + 1;
    size_t *mask = (size_t *)aligned_alloc(align_size, mask_size * sizeof(size_t));
    memset(mask, 0, mask_size * sizeof(size_t));
    int stidx = get_start_index(sellcs_matrix.row_order, sellcs_matrix.nrows);
    int pos = stidx / 64;
    size_t t = 1;
    mask[pos] = (t << (63 - (stidx % 64)));

    int x_max = 0;
    int x_min = sellcs_matrix.nrows;
    struct timeval t_start, t_end;
    double t_merge = 0;
    double t_check = 0;
    double t_kernel = 0;
    double t_pre = 0;
    double t_scan = 0;
    double t_memset = 0;
    char *mode;
    int bypass_slice = 0;
    int bypass_width = 0;
    int *s_bypass = (int *)malloc(sellcs_matrix.nrows * sizeof(int));
    memset(s_bypass, 0, sellcs_matrix.nrows * sizeof(int));
    int *w_bypass = (int *)malloc(sellcs_matrix.nrows * sizeof(int));
    memset(w_bypass, 0, sellcs_matrix.nrows * sizeof(int));

    int *arr_xmax = (int *)malloc(sellcs_matrix.nrows * sizeof(int));
    memset(arr_xmax, 0, sellcs_matrix.nrows * sizeof(int));
    int *arr_xmin = (int *)malloc(sellcs_matrix.nrows * sizeof(int));
    memset(arr_xmin, 0, sellcs_matrix.nrows * sizeof(int));

    int nwidth = sellcs_matrix.vop_pointers[sellcs_matrix.nslices];
    chunk_size = 8;
    int *task_sizes = get_task_ptrs(&sellcs_matrix, chunk_size);
    int *thread_y_nnz = (int *)malloc((chunk_size + 1) * sizeof(int));
    memset(thread_y_nnz, 0, (chunk_size + 1) * sizeof(int));

    int MAX_threads = omp_get_max_threads();

    int *compressed_x = (int *)malloc(((sellcs_matrix.nrows / 256) + 1) * sizeof(int));
    memset(compressed_x, 0, ((sellcs_matrix.nrows / 256) + 1) * sizeof(int));
    compressed_x[0] = 1;

    // Warm up
    for (int i = 0; i < 50; i++)
    {
#pragma omp parallel for
        for (int i = 0; i < chunk_size; i++)
        {
            int slice_idx = task_sizes[i];
            int end_slice = task_sizes[i + 1];
            kernel_SELLCS_DFC(&sellcs_matrix, bfs_x, bfs_y, slice_idx, end_slice);
        }
    }
    memset(bfs_y, 0, csr_matrix->nrows * sizeof(elem_t));
    int *merge_row_order = (int *)malloc(csr_matrix->nrows * sizeof(int));
    memset(merge_row_order, 0, csr_matrix->nrows * sizeof(int));

    int *vop_length_ptr = (int *)malloc((nwidth + 1) * sizeof(int));
    memcpy(vop_length_ptr, sellcs_matrix.vop_lengths, nwidth * sizeof(int));
    exclusive_scan(vop_length_ptr, nwidth + 1);

    for (int i = 0; i < csr_matrix->nrows; i++)
    {
        int pos = sellcs_matrix.row_order[i];
        merge_row_order[pos] = i;
    }

    int *s_bypass_cnt = (int *)malloc((chunk_size + 1) * sizeof(int));
    int *w_bypass_cnt = (int *)malloc((chunk_size + 1) * sizeof(int));
    memset(s_bypass_cnt, 0, (chunk_size + 1) * sizeof(int));
    memset(w_bypass_cnt, 0, (chunk_size + 1) * sizeof(int));

    const char *last_slash_ptr = strrchr(csr_matrix->name, '/'); // Find the last occurrence of '/'

    // Calculate the length of the substring
    int substring_length = strlen(last_slash_ptr + 1);
    // Allocate memory for the character array
    char char_array[substring_length + 1];
    // Copy the substring after the last '/' into the character array
    strcpy(char_array, last_slash_ptr + 1);

    int *col_y_tmp = ((int *)malloc(chunk_size * sellcs_matrix.nrows * sizeof(int)));
    memset(col_y_tmp, 0, chunk_size * sellcs_matrix.nrows * sizeof(int));
    size_t *mask_tmp = (size_t *)aligned_alloc(align_size, (sellcs_matrix.nrows) * sizeof(size_t));
    memset(mask_tmp, 0, (sellcs_matrix.nrows) * sizeof(size_t));
    // mask[stidx] = 1;
    mask_tmp[0] = 1; //(t << (63));

    int *merge_res_tmp = (int *)malloc(sellcs_matrix.nrows * sizeof(int));
    memset(merge_res_tmp, 0, sellcs_matrix.nrows * sizeof(int));

    // build csc matrix

    SparseMatrixCSC csc_matrix;
    convert_csr_to_csc(csr_matrix, &csc_matrix);

    int SWITCH_TO_SPMSPV; // sellcs_matrix.nrows * 0.0002;//sellcs_matrix.nrows * 0.004;
    int SWITCH_TO_SPMV;

    if (bypass_record == 0){
    SWITCH_TO_SPMSPV = sellcs_matrix.nrows * 0.04; // sellcs_matrix.nrows * 0.0002;//sellcs_matrix.nrows * 0.004;
    SWITCH_TO_SPMV = sellcs_matrix.nrows * 0.04;}
    else if (bypass_record == 1){
    SWITCH_TO_SPMSPV = sellcs_matrix.nrows; 
    SWITCH_TO_SPMV = sellcs_matrix.nrows;}
    else{
    SWITCH_TO_SPMSPV = 0; 
    SWITCH_TO_SPMV = 0;}

 

    gettimeofday(&t_start, NULL);

    do
    {
        if (perf_analysis)
        {
            gettimeofday(&t1, NULL);
        }
        if ((*nnzx) <= SWITCH_TO_SPMSPV)
        {
            x_max = 0;
            x_min = sellcs_matrix.nrows;
            for (int i = 0; i < (*nnzx); i++)
            {
                x_max = x_max < colidx_x[i] ? colidx_x[i] : x_max;
                x_min = x_min > colidx_x[i] ? colidx_x[i] : x_min;
            }
#pragma omp parallel for
            for (int i = 0; i < (*nnzx); i++)
            {
                int pos = colidx_x[i];
                bucket[pos] = i;
            }
        }
        if (perf_analysis)
        {
            gettimeofday(&t2, NULL);
            t_pre += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }

#if defined(USE_OMP)
        memset(thread_y_nnz, 0, (chunk_size + 1) * sizeof(int));
        if (perf_analysis)
        {
            gettimeofday(&t1, NULL);
        }

#pragma omp parallel for
        for (int i = 0; i < chunk_size; i++)
        {
            int slice_idx = task_sizes[i];
            int end_slice = task_sizes[i + 1];
            int nnz_perthread = 0;
            int th_bypass_slice = 0;
            int th_bypass_width = 0;
            if (task_sizes[i] < sellcs_matrix.nslices)
            {
                // kernel_SELLCS_U8_NC_DFC(&sellcs_matrix, bfs_x, bfs_y, slice_idx, end_slice);
                if (bit_flag)
                {
                    if ((*nnzx) > SWITCH_TO_SPMV)
                    {
                        mode = "spmv";
                        bfs_kernel_SELLCS_DFC_bitmap(&sellcs_matrix,
                                                     mask_x,
                                                     mask_y,
                                                     &nnz_perthread,
                                                     slice_idx,
                                                     end_slice,
                                                     mask,
                                                     &th_bypass_slice);
                    }
                    else if ((*nnzx) <= SWITCH_TO_SPMSPV)
                    {
                        mode = "spmspv";
                        bfs_kernel_spmspv_SELLCS_DFC_bitmap(&sellcs_matrix,
                                                            slice_idx_max,
                                                            slice_idx_min,
                                                            bucket,
                                                            colidx_x,
                                                            nnzx,
                                                            x_max, x_min,
                                                            colidx_y,
                                                            &nnz_perthread,
                                                            slice_idx,
                                                            end_slice,
                                                            mask,
                                                            &th_bypass_slice,
                                                            &th_bypass_width);
                    }
                    else
                    {

                        mode = "merge";
                        int x_per_thread = ((*nnzx) + chunk_size - 1) / chunk_size;
                        int x_start = i * x_per_thread;
                        int x_end = (i + 1) * x_per_thread > (*nnzx) ? (*nnzx) : (i + 1) * x_per_thread;
                        if (x_start < x_end)
                        {
                            csc_spmspv_merge(&csc_matrix,
                                             &sellcs_matrix,
                                             vop_length_ptr,
                                             colidx_x,
                                             nnzx,
                                             bfs_y,
                                             col_y_tmp,
                                             i * sellcs_matrix.nrows,
                                             mask_y,
                                             &nnz_perthread,
                                             merge_row_order,
                                             x_start,
                                             x_end,
                                             mask);
                        }
                    }
                }
                else
                {
                    if ((*nnzx) > SWITCH_TO_SPMV)
                    {
                        bfs_kernel_spmv_SELLCS_NC_DFC(&sellcs_matrix,
                                                      bfs_x,
                                                      bfs_y,
                                                      &nnz_perthread,
                                                      slice_idx,
                                                      end_slice,
                                                      mask);
                    }
                    else if ((*nnzx) <= SWITCH_TO_SPMSPV)
                    {
                        bfs_kernel_spmspv_SELLCS_DFC(&sellcs_matrix,
                                                     slice_idx_max,
                                                     slice_idx_min,
                                                     bucket,
                                                     bfs_x,
                                                     colidx_x,
                                                     nnzx,
                                                     x_max, x_min,
                                                     bfs_y,
                                                     colidx_y,
                                                     &nnz_perthread,
                                                     slice_idx,
                                                     end_slice,
                                                     mask,
                                                     &th_bypass_slice,
                                                     &th_bypass_width);
                    }
                }
                thread_y_nnz[i] = nnz_perthread;
                if (bypass_record != 0){
                s_bypass_cnt[i] = th_bypass_slice;
                w_bypass_cnt[i] = th_bypass_width;}
            }
        }
        if (perf_analysis)
        {
            gettimeofday(&t2, NULL);
            iter_times[(*num_iter)] = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            t_kernel += iter_times[(*num_iter)];
        }
        if (verify)
        {
            frontier[(*num_iter)] = *nnzx;
        }
        // printf("bypass slice = %lu, bypass width = %lu\n", bypass_slice, bypass_width);
        if (perf_analysis)
        {
            gettimeofday(&t1, NULL);
        }

        exclusive_scan(thread_y_nnz, chunk_size + 1);
        if (bypass_record != 0){
        exclusive_scan(s_bypass_cnt, chunk_size + 1);
        exclusive_scan(w_bypass_cnt, chunk_size + 1);
        s_bypass[*num_iter] = s_bypass_cnt[chunk_size];
        w_bypass[*num_iter] = w_bypass_cnt[chunk_size];}

        if (perf_analysis)
        {
            gettimeofday(&t2, NULL);
            t_scan += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }
        nnzy = thread_y_nnz[chunk_size];

        if (perf_analysis)
        {
            gettimeofday(&t1, NULL);
        }
        if ((*nnzx) > SWITCH_TO_SPMV) // if spmv in this iteration
        {
            if (nnzy > SWITCH_TO_SPMV) // will still be spmv in next iteration
            {
#pragma omp parallel for
                for (int i = 0; i < chunk_size; i++)
                {
                    int tasks = (sellcs_matrix.nrows + chunk_size - 1) / chunk_size;
                    int start = i * tasks;
                    int end = (i + 1) * tasks > sellcs_matrix.nrows ? sellcs_matrix.nrows : (i + 1) * tasks;
                    spmv_merge_res(bfs_x, mask_x, start, end,
                                   bfs_y, mask_y, sellcs_matrix.C, bit_flag);
                }
            }
            else
            { // will switch to spmspv in next iteration
#pragma omp parallel for

                for (int i = 0; i < chunk_size; i++)
                {
                    int slice_idx = task_sizes[i];
                    int end_slice = task_sizes[i + 1];
                    int end_pos = (end_slice << 8) > sellcs_matrix.nrows ? sellcs_matrix.nrows : (end_slice << 8);

                    dense2sparse(bfs_y, mask_y, slice_idx << 8, end_pos, thread_y_nnz[i],
                                 bfs_x, mask_x, colidx_x,
                                 sellcs_matrix.C, bit_flag, sellcs_matrix.nrows);
                }
            }
        }
        else if ((*nnzx) <= SWITCH_TO_SPMSPV)
        {                              // spmspv in this iteration
            if (nnzy > SWITCH_TO_SPMV) // will switch to spmv in next iteration
            {
#pragma omp parallel for
                for (int i = 0; i < chunk_size; i++)
                {
                    int slice_idx = task_sizes[i];
                    sparse2dense(&bfs_y[(slice_idx << 8)], &colidx_y[(slice_idx << 8)], thread_y_nnz[i + 1] - thread_y_nnz[i],
                                 bfs_x, mask_x,
                                 sellcs_matrix.C, bit_flag, sellcs_matrix.nrows);
                }
            }
            else
            { // will still be spmspv in next iteration
#pragma omp parallel for
                for (int i = 0; i < chunk_size; i++)
                {
                    int slice_idx = task_sizes[i];
                    merge_res(bfs_x, colidx_x, thread_y_nnz[i], thread_y_nnz[i + 1] - thread_y_nnz[i],
                              &bfs_y[(slice_idx << 8)], &colidx_y[(slice_idx << 8)], sellcs_matrix.C, bit_flag);
                }
            }
        }
        else
        { // merge

            int *col_y_tmp_tmp = &col_y_tmp[0];
            int nnz_tmp = thread_y_nnz[1];
            int pos;
            int slice_in_mask;
            int inner_in_mask;
            int next_res_size = 0;
            // copy the first array to the result
            for (int i = 0; i < nnz_tmp; i++)
            {
                int a = col_y_tmp_tmp[i];
                colidx_x[i] = a;
                pos = merge_row_order[a];
                slice_in_mask = pos / sellcs_matrix.C;
                inner_in_mask = pos % sellcs_matrix.C;
                int mask_pos = slice_in_mask * 4 + inner_in_mask / 64;
                mask[mask_pos] |= (1UL << (63 - (inner_in_mask % 64)));
            }

            for (int i = 1; i < chunk_size; i++)
            {
                col_y_tmp_tmp = &col_y_tmp[i * sellcs_matrix.nrows];
                int arrsize = thread_y_nnz[i + 1] - thread_y_nnz[i];
                if (arrsize > 0)
                {
                    merge_arr(colidx_x, col_y_tmp_tmp, merge_res_tmp,
                              nnz_tmp, arrsize, &next_res_size,
                              mask, merge_row_order, sellcs_matrix.C);
                    nnz_tmp = next_res_size;
                    for (int j = 0; j < next_res_size; j++)
                    {
                        colidx_x[j] = merge_res_tmp[j];
                    }
                }
            }

            nnzy = nnz_tmp;
            for (int i = 0; i < nnzy; i++)
            {
                int a = colidx_x[i];
                pos = merge_row_order[a];
                slice_in_mask = pos / sellcs_matrix.C;
                inner_in_mask = pos % sellcs_matrix.C;
                int mask_pos = slice_in_mask * 4 + inner_in_mask / 64;
                mask[mask_pos] |= (1UL << (63 - (inner_in_mask % 64)));
            }

            if (nnzy > SWITCH_TO_SPMV)
            {
                sparse2dense(&bfs_y[0], &colidx_x[0], nnzy,
                             bfs_x, mask_x,
                             sellcs_matrix.C, bit_flag, sellcs_matrix.nrows);
            }
        }
        // printf("parallel %d iters, mode = %s, nnz = %lu, nnzy = %i, time = %.3f ms\n",(*num_iter), mode, (*nnzx), nnzy, iter_times[(*num_iter)]);

        (*nnzx) = nnzy;

        if (perf_analysis)
        {
            gettimeofday(&t2, NULL);
            t_merge += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }
#else

        if (bit_flag)
        {
            // bypass_slice = 0;
            // bypass_width = 0;
            if ((*nnzx) > SWITCH) // spmv kernel
            {
                mode = "spmv";
                bfs_kernel_SELLCS_DFC_bitmap(&sellcs_matrix,
                                             mask_x,
                                             mask_y,
                                             &nnzy,
                                             0,
                                             sellcs_matrix.nslices,
                                             mask);
            }
            else
            {
                mode = "spmspv";
                bfs_kernel_spmspv_SELLCS_DFC_bitmap(&sellcs_matrix,
                                                    slice_idx_max,
                                                    slice_idx_min,
                                                    bucket,
                                                    colidx_x,
                                                    nnzx,
                                                    x_max, x_min,
                                                    colidx_y,
                                                    &nnzy,
                                                    0,
                                                    sellcs_matrix.nslices,
                                                    mask,
                                                    &bypass_slice,
                                                    &bypass_width);
            }
            // s_bypass[(*num_iter)] = bypass_slice;
            // w_bypass[(*num_iter)] = bypass_width;
        }
        else
        {

            gettimeofday(&t1, NULL);
            if ((*nnzx) > SWITCH)
            {
                // kernel_spmv_SELLCS_U8_NC_DFC(&sellcs_matrix,
                //                              bfs_x,
                //                              bfs_y,
                //                              &nnzy,
                //                              0,
                //                              sellcs_matrix.nslices,
                //                              mask);
                bfs_kernel_spmv_SELLCS_NC_DFC(&sellcs_matrix,
                                              bfs_x,
                                              bfs_y,
                                              &nnzy,
                                              0,
                                              sellcs_matrix.nslices,
                                              mask);
            }
            else
            {
                // bypass_slice = 0;
                // bypass_width = 0;

                bfs_kernel_spmspv_SELLCS_DFC(&sellcs_matrix,
                                             slice_idx_max,
                                             slice_idx_min,
                                             bucket,
                                             bfs_x,
                                             colidx_x,
                                             nnzx,
                                             x_max, x_min,
                                             bfs_y,
                                             colidx_y,
                                             &nnzy,
                                             0,
                                             sellcs_matrix.nslices,
                                             mask,
                                             &bypass_slice,
                                             &bypass_width);
                // s_bypass[(*num_iter)] = bypass_slice;
                // w_bypass[(*num_iter)] = bypass_width;
            }
        }
        if (perf_analysis)
        {
            gettimeofday(&t2, NULL);
            iter_times[(*num_iter)] = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            t_kernel += iter_times[(*num_iter)];
        }
        printf("Serial, %d iters, mode = %s, nnz = %lu, nnzy = %i, time = %.3f ms\n", (*num_iter), mode, (*nnzx), nnzy, iter_times[(*num_iter)]);

        if (verify)
        {
            frontier[(*num_iter)] = *nnzx;
        }
        // printf("bypass slice = %lu, bypass width = %lu\n", bypass_slice, bypass_width);
        if (perf_analysis)
        {
            gettimeofday(&t1, NULL);
        }

        if ((*nnzx) > SWITCH) // if spmv in this iteration
        {
            if (nnzy > SWITCH) // will still be spmv in next iteration
            {
                spmv_write_back_x(bfs_x, mask_x, nnzx,
                                  bfs_y, mask_y, nnzy,
                                  sellcs_matrix.nrows, sellcs_matrix.C, bit_flag);
            }
            else
            { // will switch to spmspv in next iteration
                dense2sparse(bfs_y, mask_y, 0, sellcs_matrix.nrows, 0,
                             bfs_x, mask_x, colidx_x,
                             sellcs_matrix.C, bit_flag, sellcs_matrix.nrows);
            }
        }
        else
        {                      // spmspv in this iteration
            if (nnzy > SWITCH) // will switch to spmv in next iteration
            {
                sparse2dense(bfs_y, colidx_y, nnzy,
                             bfs_x, mask_x,
                             sellcs_matrix.C, bit_flag, sellcs_matrix.nrows);
            }
            else
            { // will still be spmspv in next iteration
                spmspv_write_back_x(bfs_x, colidx_x, nnzx,
                                    bfs_y, colidx_y, nnzy,
                                    sellcs_matrix.C, bit_flag);
            }
        }
        if (perf_analysis)
        {
            gettimeofday(&t2, NULL);
            t_merge += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }
        (*nnzx) = nnzy;

#endif // USE_OMP

        if (perf_analysis)
        {
            gettimeofday(&t1, NULL);
        }
        if(bypass_record != 0){
        memset(s_bypass_cnt, 0, (chunk_size + 1) * sizeof(int));
        memset(w_bypass_cnt, 0, (chunk_size + 1) * sizeof(int));}

        // memset(bfs_y, 0, csr_matrix->nrows * sizeof(elem_t));
        // memset(colidx_y, 0, csr_matrix->nrows * sizeof(int));
        if (*nnzx > SWITCH_TO_SPMV)
        {
            int nnz_per_thread = (sellcs_matrix.nrows + chunk_size - 1) / chunk_size;
// memset(mask_y, 0, sellcs_matrix.nrows * sizeof(int));
#pragma omp parallel for

            for (int i = 0; i < chunk_size; i++)
            {
                int end = (i + 1) * nnz_per_thread > sellcs_matrix.nrows ? sellcs_matrix.nrows : (i + 1) * nnz_per_thread;
                reset(mask_y, i * nnz_per_thread, end, sellcs_matrix.C);
            }
        }
        (*num_vex) += (*nnzx);
        (*num_iter)++;
        if (perf_analysis)
        {
            gettimeofday(&t2, NULL);
            t_memset += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        }

    } while ((*nnzx) != 0);

    gettimeofday(&t_end, NULL);
    (*bfs_time) = (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;

    if (perf_analysis)
    {
        printf("Kernel time = %.3f ms, Merge time = %.3f ms, Preprocessing time = %.3f ms, Scan time = %.3f ms, Memset time = %.3f ms\n", t_kernel, t_merge, t_pre, t_scan, t_memset);
    }
    printf("iteration = %lu, bfs runtime = %.3f ms\n", *num_iter, *bfs_time);

    if (bypass_record == 1){
    FILE *filename = fopen("../bfs/data/bfs-slice-bypass.csv", "a");

    fprintf(filename, "%s,%i,%i,%i,%i,%i",
                      char_array, csr_matrix->nrows, csr_matrix->ncolumns, csr_matrix->nnz, *num_iter,
                      sellcs_matrix.nslices);

    for (int i = 0; i < *num_iter; i ++)
    {
        // if (i < 200){
            fprintf(filename, ",%i", s_bypass[i]);
        // }
    }
    fprintf(filename, "\n");
    fclose(filename);

    FILE *filename2 = fopen("../bfs/data/bfs-column-bypass.csv", "a");

    fprintf(filename2, "%s,%i,%i,%i,%i,%i",
                      char_array, csr_matrix->nrows, csr_matrix->ncolumns, csr_matrix->nnz, *num_iter,
                      nwidth);

    for (int i = 0; i < *num_iter; i ++)
    {
        fprintf(filename2, ",%i", w_bypass[i]);
    }
    fprintf(filename2, "\n");
    fclose(filename2);
    }
    else if (bypass_record == 2)
    {
        FILE *filename = fopen("../bfs/data/spmv-slice-bypass.csv", "a");

        fprintf(filename, "%s,%i,%i,%i,%i,%i",
                        char_array, csr_matrix->nrows, csr_matrix->ncolumns, csr_matrix->nnz, *num_iter,
                        sellcs_matrix.nslices);

        for (int i = 0; i < *num_iter; i ++)
        {
            // if (i < 200){
                fprintf(filename, ",%i", s_bypass[i]);
            // }
        }
        fprintf(filename, "\n");
        fclose(filename);
    }

}

void print_additional_custom_report(char *text_padding, double *elapsed_times)
{
    printf("%s\"Row order sigma window\": %lu,\n", text_padding, sigma_window);
    printf("%s\"Task Size\": %lu\n", text_padding, chunk_size);
}
