#include <velintrin.h>
#include "common.h"
#include "bfs_par_kernels.h"
#include "mytimer.h"
#include "utils.h"
#include "hw_counters_ve.h"

void kernel_SELLCS_DFC(const SparseMatrixSELLCS *restrict matrix,
                       const elem_t *restrict x,
                       elem_t *restrict y,
                       const int start_slice,
                       const int end_slice)
{
    const uint32_t vlen = matrix->C;

    for (int slice_idx = start_slice; slice_idx < end_slice; slice_idx++)
    {
        int row_idx = slice_idx << 8;
        uint32_t max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;

        elem_t *values_pointer = &matrix->values[matrix->slice_pointers[slice_idx]];
        int *colidx_pointer = &matrix->column_indices[matrix->slice_pointers[slice_idx]];

        __vr tmp_results = _vel_vxor_vvvl(tmp_results, tmp_results, max_lanes);

        int swidth = matrix->slice_widths[slice_idx];
        int act_lanes_idx = matrix->vop_pointers[slice_idx];

        __vr y_sc_addr = _vel_vldlzx_vssl(4, &matrix->row_order[row_idx], max_lanes);
        y_sc_addr = _vel_vsfa_vvssl(y_sc_addr, 3UL, (unsigned long)y, vlen);
        // fprintf(stderr, "Starting Slice %" PRIu64 ", w/ width %" PRIu64 "   \n", slice_idx, swidth);

        for (int i = 0; i < swidth; i++)
        {
            uint32_t act_lanes = (uint32_t)matrix->vop_lengths[act_lanes_idx++] + 1;
            // Load Values and Column indices
            __vr values_vblock = _vel_vld_vssl(8, values_pointer, act_lanes);
            __vr col_index_vblock = _vel_vldlzx_vssl(4, colidx_pointer, act_lanes);

            // Gather X
            __vr x_gt_addr = _vel_vsfa_vvssl(col_index_vblock, 3UL, (unsigned long)x, act_lanes);
            __vr x_vblock = _vel_vgt_vvssl(x_gt_addr, (uint64_t)&x[0], (uint64_t)&x[matrix->ncolumns + 1], act_lanes);

            // Multiply
            tmp_results = _vel_vfmadd_vvvvl(tmp_results, x_vblock, values_vblock, act_lanes);

            values_pointer += act_lanes;
            colidx_pointer += act_lanes;
        }

        _vel_vsc_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], max_lanes);
        // _vel_vscncot_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], act_lanes);
    }
}
void bfs_kernel_spmv_SELLCS_NC_DFC(const SparseMatrixSELLCS *restrict matrix,
                                   const elem_t *x,
                                   elem_t *y,
                                   int *nnzy,
                                   const int start_slice,
                                   const int end_slice,
                                   size_t *mask)
{
    const uint32_t vlen = matrix->C;
    int nnz_tmp = 0;

    for (int slice_idx = start_slice; slice_idx < end_slice; slice_idx++)
    {
        int row_idx = slice_idx << 8;
        uint32_t max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;
        // load y_mask
        __vr check_vblock = _vel_vld_vssl(8, &mask[row_idx], max_lanes);
        __vm256 check_mask = _vel_vfmklgt_mvl(check_vblock, max_lanes);

        unsigned long int nnz = _vel_pcvm_sml(check_mask, max_lanes);
        if (nnz < max_lanes)
        {

            int elem_idx = matrix->slice_pointers[slice_idx];
            elem_t *values_pointer = &matrix->values[elem_idx];
            int *colidx_pointer = &matrix->column_indices[elem_idx];

            __vr tmp_results = _vel_vxor_vvvl(tmp_results, tmp_results, max_lanes);

            int swidth = matrix->slice_widths[slice_idx];
            int act_lanes_idx = matrix->vop_pointers[slice_idx];
            __vr y_sc_addr = _vel_vldlzx_vssl(4, &matrix->row_order[row_idx], max_lanes);
            y_sc_addr = _vel_vsfa_vvssl(y_sc_addr, 3UL, (unsigned long)y, max_lanes);
            // fprintf(stderr, "Starting Slice %" PRIu64 ", w/ width %" PRIu64 "   \n", slice_idx, swidth);

            for (int i = 0; i < swidth; i++)
            {
                uint32_t act_lanes = (uint32_t)matrix->vop_lengths[act_lanes_idx++] + 1;
                // Load Values and Column indices
                __vr values_vblock = _vel_vldnc_vssl(8, values_pointer, act_lanes);
                __vr col_index_vblock = _vel_vldlzx_vssl(4, colidx_pointer, act_lanes);
                //     // Gather X
                __vr x_gt_addr = _vel_vsfa_vvssl(col_index_vblock, 3UL, (unsigned long)x, act_lanes);
                __vr x_vblock = _vel_vgt_vvssl(x_gt_addr, (uint64_t)&x[0], (uint64_t)&x[matrix->ncolumns + 1], act_lanes);
                // // Multiply
                tmp_results = _vel_vfmadd_vvvvl(tmp_results, x_vblock, values_vblock, act_lanes);

                values_pointer += act_lanes;
                colidx_pointer += act_lanes;
            }
            __vm256 results_mask = _vel_vfmklaf_ml(vlen);
            results_mask = _vel_vfmkdgt_mvl(tmp_results, max_lanes);
            results_mask = _vel_nndm_mmm(check_mask, results_mask);
            check_mask = _vel_orm_mmm(check_mask, results_mask);

            // mask[slice_idx * 4] = _vel_svm_sms(check_mask, 0);
            // mask[slice_idx * 4 + 1] = _vel_svm_sms(check_mask, 1);
            // mask[slice_idx * 4 + 2] = _vel_svm_sms(check_mask, 2);
            // mask[slice_idx * 4 + 3] = _vel_svm_sms(check_mask, 3);

            check_vblock = _vel_vor_vsvmvl(1UL, check_vblock, check_mask, check_vblock, max_lanes);
            _vel_vst_vssl(check_vblock, 8, &mask[row_idx], max_lanes);

            unsigned long int slice_nnz = _vel_pcvm_sml(results_mask, max_lanes);
            nnz_tmp += slice_nnz;
            _vel_vscot_vvssml(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], results_mask, max_lanes);
        }
        // _vel_vsc_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], max_lanes);
        // _vel_vscncot_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], act_lanes);
    }
    (*nnzy) = nnz_tmp;
}

void bfs_kernel_SELLCS_DFC_bitmap(const SparseMatrixSELLCS *restrict matrix,
                                  const int *mask_x,
                                  int *mask_y,
                                  int *nnzy,
                                  const int start_slice,
                                  const int end_slice,
                                  size_t *mask,
                                  int *bypass_slice)
{
    const uint32_t vlen = matrix->C;
    int nnz_tmp = 0;

    for (int slice_idx = start_slice; slice_idx < end_slice; slice_idx++)
    {
        int row_idx = slice_idx << 8;
        uint32_t max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;

        __vr tmp_results = _vel_vxor_vvvl(tmp_results, tmp_results, max_lanes);

        __vm256 check_mask = _vel_vfmklaf_ml(max_lanes);
        check_mask = _vel_lvm_mmss(check_mask, 0, (uint64_t)mask[slice_idx * 4]);
        check_mask = _vel_lvm_mmss(check_mask, 1, (uint64_t)mask[slice_idx * 4 + 1]);
        check_mask = _vel_lvm_mmss(check_mask, 2, (uint64_t)mask[slice_idx * 4 + 2]);
        check_mask = _vel_lvm_mmss(check_mask, 3, (uint64_t)mask[slice_idx * 4 + 3]);
        unsigned long int nnz = _vel_pcvm_sml(check_mask, max_lanes);
        if (nnz < max_lanes)
        {
            int *colidx_pointer = &matrix->column_indices[matrix->slice_pointers[slice_idx]];

            int swidth = matrix->slice_widths[slice_idx];
            int act_lanes_idx = matrix->vop_pointers[slice_idx];

            __vr y_sc_addr = _vel_vldlzx_vssl(4, &matrix->row_order[row_idx], max_lanes);
            y_sc_addr = _vel_vsfa_vvssl(y_sc_addr, 2UL, (unsigned long)mask_y, vlen);
            // fprintf(stderr, "Starting Slice %" PRIu64 ", w/ width %" PRIu64 "   \n", slice_idx, swidth);

            for (int i = 0; i < swidth; i++)
            {
                uint32_t act_lanes = (uint32_t)matrix->vop_lengths[act_lanes_idx++] + 1;
                // Load Column indices
                __vr col_index_vblock = _vel_vldlzx_vssl(4, colidx_pointer, act_lanes);
                // Gather mask_x
                __vr x_gt_addr = _vel_vsfa_vvssl(col_index_vblock, 2UL, (unsigned long)mask_x, act_lanes);
                __vr mask_x_vblock = _vel_vgtlzx_vvssl(x_gt_addr, (uint64_t)&mask_x[0], (uint64_t)&mask_x[matrix->ncolumns + 1], act_lanes);
                // OR
                tmp_results = _vel_vor_vvvl(tmp_results, mask_x_vblock, act_lanes);
                colidx_pointer += act_lanes;
            }
            __vm256 results_mask = _vel_vfmklaf_ml(max_lanes);
            results_mask = _vel_vfmklgt_mvl(tmp_results, max_lanes);
            results_mask = _vel_nndm_mmm(check_mask, results_mask);
            check_mask = _vel_orm_mmm(check_mask, results_mask);

            mask[slice_idx * 4] = _vel_svm_sms(check_mask, 0);
            mask[slice_idx * 4 + 1] = _vel_svm_sms(check_mask, 1);
            mask[slice_idx * 4 + 2] = _vel_svm_sms(check_mask, 2);
            mask[slice_idx * 4 + 3] = _vel_svm_sms(check_mask, 3);

            unsigned long int slice_nnz = _vel_pcvm_sml(results_mask, max_lanes);
            nnz_tmp += slice_nnz;
            _vel_vscl_vvssml(tmp_results, y_sc_addr, (uint64_t)&mask_y[0], (uint64_t)&mask_y[matrix->nrows], results_mask, max_lanes);
        }
        else
        {
            (*bypass_slice)++;
        }
        // _vel_vsc_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], max_lanes);
        // _vel_vscncot_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], act_lanes);
    }
    (*nnzy) = nnz_tmp;
}

void bfs_kernel_SELLCS_DFC_bitmap_v2(const SparseMatrixSELLCS *restrict matrix,
                                     const int *mask_x,
                                     int *mask_y,
                                     int *nnzy,
                                     const int start_slice,
                                     const int end_slice,
                                     size_t *mask,
                                     int *compressed_x)

{
    const uint32_t vlen = matrix->C;
    int nnz_tmp = 0;

    for (int slice_idx = start_slice; slice_idx < end_slice; slice_idx++)
    {
        int row_idx = slice_idx << 8;
        uint32_t max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;

        __vr tmp_results = _vel_vxor_vvvl(tmp_results, tmp_results, max_lanes);

        __vm256 check_mask = _vel_vfmklaf_ml(max_lanes);
        check_mask = _vel_lvm_mmss(check_mask, 0, (uint64_t)mask[slice_idx * 4]);
        check_mask = _vel_lvm_mmss(check_mask, 1, (uint64_t)mask[slice_idx * 4 + 1]);
        check_mask = _vel_lvm_mmss(check_mask, 2, (uint64_t)mask[slice_idx * 4 + 2]);
        check_mask = _vel_lvm_mmss(check_mask, 3, (uint64_t)mask[slice_idx * 4 + 3]);
        unsigned long int nnz = _vel_pcvm_sml(check_mask, max_lanes);
        // printf("nnz before = %li\n", nnz);
        if (nnz < max_lanes)
        {
            int *colidx_pointer = &matrix->column_indices[matrix->slice_pointers[slice_idx]];

            int swidth = matrix->slice_widths[slice_idx];
            int act_lanes_idx = matrix->vop_pointers[slice_idx];

            __vr y_sc_addr = _vel_vldlzx_vssl(4, &matrix->row_order[row_idx], max_lanes);
            y_sc_addr = _vel_vsfa_vvssl(y_sc_addr, 2UL, (unsigned long)mask_y, vlen);
            // fprintf(stderr, "Starting Slice %" PRIu64 ", w/ width %" PRIu64 "   \n", slice_idx, swidth);

            for (int i = 0; i < swidth; i++)
            {
                uint32_t act_lanes = (uint32_t)matrix->vop_lengths[act_lanes_idx++] + 1;
                // Load Column indices
                __vr col_index_vblock = _vel_vldlzx_vssl(4, colidx_pointer, act_lanes);

                // gather compressed x firstly
                __vr vcompressed_tmp = _vel_vdivswzx_vvsl(col_index_vblock, 256, act_lanes);
                // __vr vcompressed_tmp = _vel_pvsra_vvsl(col_index_vblock, 8UL, act_lanes);
                __vr compressed_addr = _vel_vsfa_vvssl(vcompressed_tmp, 2UL, (unsigned long)compressed_x, act_lanes);
                __vr vcompressed_x = _vel_vgtlzx_vvssl(compressed_addr, (uint64_t)&compressed_x[0], (uint64_t)&compressed_x[((matrix->ncolumns) / 256) + 1] + 1, act_lanes);
                __vm256 vmask_compress = _vel_vfmklgt_mvl(vcompressed_x, act_lanes);
                unsigned long compress_nnz = _vel_pcvm_sml(vmask_compress, act_lanes);
                // printf("nnz = %li\n", compress_nnz);
                if (compress_nnz != 0)
                {

                    // Gather mask_x
                    __vr x_gt_addr = _vel_vsfa_vvssl(col_index_vblock, 2UL, (unsigned long)mask_x, act_lanes);
                    __vr mask_x_vblock = _vel_vgtlzx_vvssl(x_gt_addr, (uint64_t)&mask_x[0], (uint64_t)&mask_x[matrix->ncolumns + 1], act_lanes);
                    // OR
                    tmp_results = _vel_vor_vvvl(tmp_results, mask_x_vblock, act_lanes);
                }
                colidx_pointer += act_lanes;
            }
            __vm256 results_mask = _vel_vfmklaf_ml(max_lanes);
            results_mask = _vel_vfmklgt_mvl(tmp_results, max_lanes);
            results_mask = _vel_nndm_mmm(check_mask, results_mask);
            check_mask = _vel_orm_mmm(check_mask, results_mask);

            mask[slice_idx * 4] = _vel_svm_sms(check_mask, 0);
            mask[slice_idx * 4 + 1] = _vel_svm_sms(check_mask, 1);
            mask[slice_idx * 4 + 2] = _vel_svm_sms(check_mask, 2);
            mask[slice_idx * 4 + 3] = _vel_svm_sms(check_mask, 3);

            unsigned long int slice_nnz = _vel_pcvm_sml(results_mask, max_lanes);
            nnz_tmp += slice_nnz;
            // if (slice_nnz != 0)
            // {
            //     compressed_x[slice_idx] = 1;
            // }
            _vel_vscl_vvssml(tmp_results, y_sc_addr, (uint64_t)&mask_y[0], (uint64_t)&mask_y[matrix->nrows], results_mask, max_lanes);
        }
        // _vel_vsc_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], max_lanes);
        // _vel_vscncot_vvssl(tmp_results, y_sc_addr, (uint64_t)&y[0], (uint64_t)&y[matrix->nrows], act_lanes);
    }
    (*nnzy) = nnz_tmp;
}

void merge_kernel(int *arr_a, int *arr_b, int *res,
                  int size_a, int size_b, int *res_size,
                  size_t *mask, int *merge_row_order, int vlen)
{
    int i = 0, j = 0, k = 0;
    int pos;
    int slice_in_mask;
    int inner_in_mask;
    unsigned long int bit_mask;
    int a, b;

    while (i < size_a && j < size_b)
    {
        a = arr_a[i];
        b = arr_b[j];
        if (a < b)
        {
            res[k++] = a;
            i++;
        }
        else if (a > b)
        {
            pos = merge_row_order[b];
            slice_in_mask = pos / vlen;
            inner_in_mask = pos % vlen;
            bit_mask = 1UL << (63 - (inner_in_mask % 64));
            if ((mask[slice_in_mask * 4 + inner_in_mask / 64] & bit_mask) >> (63 - (inner_in_mask % 64)) == 0)
            {

                res[k++] = b;
            }
            j++;
        }
        else
        {
            res[k++] = a;
            i++;
            j++;
        }
    }
    while (i < size_a)
    {
        a = arr_a[i];
        res[k++] = a;
        i++;
    }

    while (j < size_b)
    {
        b = arr_b[j];
        pos = merge_row_order[b];
        slice_in_mask = pos / vlen;
        inner_in_mask = pos % vlen;
        bit_mask = 1UL << (63 - (inner_in_mask % 64));
        if ((mask[slice_in_mask * 4 + inner_in_mask / 64] & bit_mask) >> (63 - (inner_in_mask % 64)) == 0)
        {
            res[k++] = b;
        }
        j++;
    }
    (*res_size) = k;
}

void csc_spmspv_merge(const SparseMatrixCSC *csc_matrix,
                      const SparseMatrixSELLCS *restrict matrix,
                      const int *vop_length_ptr,
                      const int *colidx_x,
                      const int *nnzx,
                      elem_t *restrict y,
                      int *colidx_y,
                      int col_y_pos,
                      int *mask_y,
                      int *nnzy,
                      int *merge_row_order,
                      int start_pos,
                      int end_pos,
                      size_t *mask)
{
    int pos;
    int slice_in_mask;
    int inner_in_mask;
    unsigned long int bit_mask;

    // copy the first array to the result
    int start_idx = colidx_x[start_pos];
    int elem_start = csc_matrix->column_pointers[start_idx];
    int elem_end = csc_matrix->column_pointers[start_idx + 1];
    int j = 0;
    int *col_y_tmp = &colidx_y[col_y_pos];
    for (int i = elem_start; i < elem_end; i++)
    {
        int colidx = csc_matrix->row_indices[i];
        pos = merge_row_order[colidx];
        slice_in_mask = pos / matrix->C;
        inner_in_mask = pos % matrix->C;
        bit_mask = 1UL << (63 - (inner_in_mask % 64));
        if ((mask[slice_in_mask * 4 + inner_in_mask / 64] & bit_mask) >> (63 - (inner_in_mask % 64)) == 0)
        {
            col_y_tmp[j++] = csc_matrix->row_indices[i];
        }
    }
    int col_y_size = j;

    if (end_pos > start_pos + 1)
    {
        int arr_size1;
        int arr_size2;
        int *next_res = (int *)malloc(matrix->nrows * sizeof(int));
        // memset(next_res, 0, matrix->nrows * sizeof(int));
        int next_res_size;
        int *column_index_ptr;
        int i = 0;
        for (i = start_pos + 1; i < end_pos; i++)
        {
            int idx = colidx_x[i];
            int arr_start = csc_matrix->column_pointers[idx];
            column_index_ptr = &csc_matrix->row_indices[arr_start];
            arr_size1 = csc_matrix->column_pointers[idx + 1] - arr_start;
            merge_kernel(col_y_tmp, column_index_ptr, next_res,
                         col_y_size, arr_size1, &next_res_size,
                         mask, merge_row_order, matrix->C);
            col_y_size = next_res_size;
            for (int k = 0; k < col_y_size; k++)
            {
                col_y_tmp[k] = next_res[k];
            }
        }
        *nnzy = col_y_size;
    }
}

void spmspv_merge(const SparseMatrixCSR *csr_matrix,
                  const SparseMatrixSELLCS *restrict matrix,
                  const int *vop_length_ptr,
                  const int *colidx_x,
                  const int *nnzx,
                  elem_t *restrict y,
                  int *colidx_y,
                  int col_y_pos,
                  int *mask_y,
                  int *nnzy,
                  int *merge_row_order,
                  int start_pos,
                  int end_pos,
                  size_t *mask)
{
    int nnz_tmp = 0;
    int *col_y_tmp = &colidx_y[col_y_pos];
    // size_t *mask_tmp_tmp = &mask[mask_pos];
    for (int i = start_pos; i < end_pos; i++)
    {
        int idx = colidx_x[i];
        for (int j = csr_matrix->row_pointers[idx]; j < csr_matrix->row_pointers[idx + 1]; j++)
        {
            int colidx = csr_matrix->column_indices[j];
            int pos_y = merge_row_order[colidx];
            int slice_in_mask = pos_y / matrix->C;
            int inner_in_mask = pos_y % matrix->C;
            unsigned long int bit_mask = 1UL << (63 - (inner_in_mask % 64));
            if ((mask[slice_in_mask * 4 + inner_in_mask / 64] & bit_mask) >> (63 - (inner_in_mask % 64)) == 0)
            {
                col_y_tmp[nnz_tmp] = colidx;
                // mask_tmp_tmp[slice_in_mask * 4 + inner_in_mask / 64] |= (1UL << (63 - (inner_in_mask % 64)));
                nnz_tmp++;
            }
        }
    }
    *nnzy = nnz_tmp;
}

void bfs_kernel_spmspv_SELLCS_DFC(const SparseMatrixSELLCS *restrict matrix,
                                  const int *slice_idx_max,
                                  const int *slice_idx_min,
                                  const int *bucket,
                                  const elem_t *restrict x,
                                  const int *colidx_x,
                                  const int *nnzx,
                                  const int x_max,
                                  const int x_min,
                                  elem_t *restrict y,
                                  int *colidx_y,
                                  int *nnzy,
                                  const int start_slice,
                                  const int end_slice,
                                  size_t *mask,
                                  int *bypass_slice,
                                  int *bypass_width)
{
    const uint32_t vlen = matrix->C;
    int nrows = matrix->nrows;
    int nnz_tmp = 0;
    for (int slice_idx = start_slice; slice_idx < end_slice; slice_idx++)
    {
        if (!(slice_idx_max[slice_idx] < x_min || slice_idx_min[slice_idx] > x_max))
        {
            int row_idx = slice_idx << 8;
            uint32_t max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;
            __vm256 check_mask = _vel_vfmklaf_ml(max_lanes);
            check_mask = _vel_lvm_mmss(check_mask, 0, (uint64_t)mask[slice_idx * 4]);
            check_mask = _vel_lvm_mmss(check_mask, 1, (uint64_t)mask[slice_idx * 4 + 1]);
            check_mask = _vel_lvm_mmss(check_mask, 2, (uint64_t)mask[slice_idx * 4 + 2]);
            check_mask = _vel_lvm_mmss(check_mask, 3, (uint64_t)mask[slice_idx * 4 + 3]);
            unsigned long int nnz = _vel_pcvm_sml(check_mask, max_lanes);
            if (nnz < max_lanes)
            {

                int elem_idx = matrix->slice_pointers[slice_idx];
                elem_t *values_pointer = &matrix->values[elem_idx];
                int *colidx_pointer = &matrix->column_indices[elem_idx];

                __vr tmp_results = _vel_vxor_vvvl(tmp_results, tmp_results, max_lanes);
                int swidth = matrix->slice_widths[slice_idx];
                int act_lanes_idx = matrix->vop_pointers[slice_idx];
                __vr y_sc_addr = _vel_vldlzx_vssl(4, &matrix->row_order[row_idx], max_lanes);
                __vm256 result_mask = _vel_vfmklaf_ml(max_lanes);
                for (int i = 0; i < swidth; i++)
                {
                    uint32_t act_lanes = (uint32_t)matrix->vop_lengths[act_lanes_idx + i] + 1;

                    if (!(matrix->max_index[act_lanes_idx + i] < x_min || matrix->min_index[act_lanes_idx + i] > x_max))
                    {
                        // Load Column indices
                        __vr col_index_vblock = _vel_vldlzx_vssl(4, colidx_pointer, act_lanes);
                        // Gather bucket
                        __vr bucket_vblock = _vel_vsfa_vvssl(col_index_vblock, 2UL, (unsigned long)bucket, act_lanes);
                        bucket_vblock = _vel_vgtlzx_vvssl(bucket_vblock, (uint64_t)&bucket[0], (uint64_t)&bucket[nrows] + 1, act_lanes);
                        // gather Colidx_x
                        __vr colidx_x_vblock = _vel_vsfa_vvssl(bucket_vblock, 2UL, (unsigned long)colidx_x, act_lanes);
                        colidx_x_vblock = _vel_vgtlzx_vvssl(colidx_x_vblock, (uint64_t)&colidx_x[0], (uint64_t)&colidx_x[(*nnzx)] + 1, act_lanes);
                        __vr sub_vblock = _vel_vsubul_vvvl(colidx_x_vblock, col_index_vblock, act_lanes);
                        // compare and get the mask
                        __vm256 column_mask = _vel_vfmkleq_mvl(sub_vblock, act_lanes);
                        unsigned long int match_cnt = _vel_pcvm_sml(column_mask, act_lanes);

                        if (match_cnt != 0)
                        {
                            // Load Values
                            __vr values_vblock = _vel_vld_vssl(8, values_pointer, act_lanes);
                            // Gather X
                            __vr x_vblock = _vel_vsfa_vvssl(bucket_vblock, 3UL, (unsigned long)colidx_x, act_lanes);
                            x_vblock = _vel_vgt_vvssml(x_vblock, (uint64_t)&x[0], (uint64_t)&x[(*nnzx)] + 1, column_mask, act_lanes);

                            // Multiply
                            tmp_results = _vel_vfmadd_vvvvmvl(tmp_results, x_vblock, values_vblock, column_mask, tmp_results, act_lanes);
                        }
                        result_mask = _vel_orm_mmm(result_mask, column_mask);
                    }
                    // else{
                    //     (bypass_width) ++;
                    // }
                    values_pointer += act_lanes;
                    colidx_pointer += act_lanes;
                }

                result_mask = _vel_nndm_mmm(check_mask, result_mask);
                check_mask = _vel_orm_mmm(check_mask, result_mask);

                mask[slice_idx * 4] = _vel_svm_sms(check_mask, 0);
                mask[slice_idx * 4 + 1] = _vel_svm_sms(check_mask, 1);
                mask[slice_idx * 4 + 2] = _vel_svm_sms(check_mask, 2);
                mask[slice_idx * 4 + 3] = _vel_svm_sms(check_mask, 3);
                unsigned long int slice_nnz = _vel_pcvm_sml(result_mask, max_lanes);

                if (slice_nnz != 0)
                {
                    __vr result_vcp = _vel_vcp_vvmvl(tmp_results, result_mask, tmp_results, max_lanes);
                    __vr colidxy_vcp = _vel_vcp_vvmvl(y_sc_addr, result_mask, y_sc_addr, max_lanes);
                    _vel_vst_vssl(result_vcp, 8, &y[(start_slice << 8) + nnz_tmp], slice_nnz);
                    _vel_vstl_vssl(colidxy_vcp, 4, &colidx_y[(start_slice << 8) + nnz_tmp], slice_nnz);
                    nnz_tmp += slice_nnz;
                }
            }
        }
        // else{
        //     (*bypass_slice) ++;
        // }
    }
    (*nnzy) = nnz_tmp;
}
void bfs_kernel_spmspv_SELLCS_DFC_bitmap(const SparseMatrixSELLCS *restrict matrix,
                                         const int *slice_idx_max,
                                         const int *slice_idx_min,
                                         const int *bucket,
                                         const int *colidx_x,
                                         const int *nnzx,
                                         const int x_max,
                                         const int x_min,
                                         int *colidx_y,
                                         int *nnzy,
                                         const int start_slice,
                                         const int end_slice,
                                         size_t *mask,
                                         int *bypass_slice,
                                         int *bypass_width)

{
    const uint32_t vlen = matrix->C;
    int nrows = matrix->nrows;
    int nnz_tmp = 0;
    for (int slice_idx = start_slice; slice_idx < end_slice; slice_idx++)
    {
        if (!(slice_idx_max[slice_idx] < x_min || slice_idx_min[slice_idx] > x_max))
        {
            int row_idx = slice_idx << 8;
            uint32_t max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;
            __vm256 check_mask = _vel_vfmklaf_ml(max_lanes);
            check_mask = _vel_lvm_mmss(check_mask, 0, (uint64_t)mask[slice_idx * 4]);
            check_mask = _vel_lvm_mmss(check_mask, 1, (uint64_t)mask[slice_idx * 4 + 1]);
            check_mask = _vel_lvm_mmss(check_mask, 2, (uint64_t)mask[slice_idx * 4 + 2]);
            check_mask = _vel_lvm_mmss(check_mask, 3, (uint64_t)mask[slice_idx * 4 + 3]);
            unsigned long int nnz = _vel_pcvm_sml(check_mask, max_lanes);

            if (nnz < max_lanes)
            {

                int elem_idx = matrix->slice_pointers[slice_idx];
                int *colidx_pointer = &matrix->column_indices[elem_idx];

                int swidth = matrix->slice_widths[slice_idx];
                int act_lanes_idx = matrix->vop_pointers[slice_idx];
                __vr y_sc_addr = _vel_vldlzx_vssl(4, &matrix->row_order[row_idx], max_lanes);

                __vm256 result_mask = _vel_vfmklaf_ml(max_lanes);
                for (int i = 0; i < swidth; i++)
                {
                    uint32_t act_lanes = (uint32_t)matrix->vop_lengths[act_lanes_idx + i] + 1;
                    if (!(matrix->max_index[act_lanes_idx + i] < x_min || matrix->min_index[act_lanes_idx + i] > x_max))
                    {
                        // Load Column indices
                        __vr col_index_vblock = _vel_vldlzx_vssl(4, colidx_pointer, act_lanes);
                        // Gather bucket
                        __vr bucket_addr = _vel_vsfa_vvssl(col_index_vblock, 2UL, (unsigned long)bucket, act_lanes);
                        __vr bucket_vblock = _vel_vgtlzx_vvssl(bucket_addr, (uint64_t)&bucket[0], (uint64_t)&bucket[nrows] + 1, act_lanes);
                        // gather Colidx_x
                        __vr colidx_x_vblock = _vel_vsfa_vvssl(bucket_vblock, 2UL, (unsigned long)colidx_x, act_lanes);
                        colidx_x_vblock = _vel_vgtlzx_vvssl(colidx_x_vblock, (uint64_t)&colidx_x[0], (uint64_t)&colidx_x[(*nnzx)] + 1, act_lanes);
                        __vr sub_vblock = _vel_vsubul_vvvl(colidx_x_vblock, col_index_vblock, act_lanes);
                        // compare and get the mask
                        __vm256 column_mask = _vel_vfmkleq_mvl(sub_vblock, act_lanes);
                        result_mask = _vel_orm_mmm(result_mask, column_mask);
                    }
                    else
                    {
                        (*bypass_width)++;
                    }
                    colidx_pointer += act_lanes;
                }
                result_mask = _vel_nndm_mmm(check_mask, result_mask);
                check_mask = _vel_orm_mmm(check_mask, result_mask);

                mask[slice_idx * 4] = _vel_svm_sms(check_mask, 0);
                mask[slice_idx * 4 + 1] = _vel_svm_sms(check_mask, 1);
                mask[slice_idx * 4 + 2] = _vel_svm_sms(check_mask, 2);
                mask[slice_idx * 4 + 3] = _vel_svm_sms(check_mask, 3);
                unsigned long int slice_nnz = _vel_pcvm_sml(result_mask, max_lanes);
                if (slice_nnz != 0)
                {
                    __vr colidxy_vcp = _vel_vcp_vvmvl(y_sc_addr, result_mask, y_sc_addr, max_lanes);
                    _vel_vstl_vssl(colidxy_vcp, 4, &colidx_y[(start_slice << 8) + nnz_tmp], slice_nnz);
                    nnz_tmp += slice_nnz;
                }
            }
        }
        else
        {
            (*bypass_slice)++;
        }
    }
    (*nnzy) = nnz_tmp;
}

void spmspv_write_back_x(elem_t *x, int *colidx_x, int *nnzx,
                         elem_t *y, int *colidx_y, int nnzy,
                         int MVL, bool bit_flag)
{
    for (int i = 0; i < nnzy; i += MVL)
    {
        int VL = (i + MVL) > nnzy ? (nnzy - i) : MVL;
        if (!bit_flag)
        {
            __vr y_vblock = _vel_vld_vssl(8, &y[i], VL);
            _vel_vst_vssl(y_vblock, 8, &x[i], VL);
        }
        __vr colidxy_vblock = _vel_vldlzx_vssl(4, &colidx_y[i], VL);
        _vel_vstlot_vssl(colidxy_vblock, 4, (uint64_t)&colidx_x[i], VL);
    }
    (*nnzx) = nnzy;
}
void spmv_write_back_x(elem_t *x, int *mask_x, int *nnzx,
                       elem_t *y, int *mask_y, int nnzy,
                       int nrows, int MVL, bool bit_flag)
{
    for (int i = 0; i < nrows; i += MVL)
    {
        int VL = (i + MVL) > nrows ? (nrows - i) : MVL;
        if (!bit_flag)
        {
            __vr y_vblock = _vel_vld_vssl(8, &y[i], VL);
            _vel_vst_vssl(y_vblock, 8, &x[i], VL);
        }
        else
        {
            __vr y_vblock = _vel_vldlzx_vssl(4, &mask_y[i], VL);
            _vel_vstlot_vssl(y_vblock, 4, (uint64_t)&mask_x[i], VL);
        }
    }
    *nnzx = nnzy;
}

void spmv_merge_res(elem_t *x, int *mask_x, int start_pos, int end_pos,
                    elem_t *y, int *mask_y, int MVL, bool bit_flag)
{
    for (int i = start_pos; i < end_pos; i += MVL)
    {
        int VL = (i + MVL) > end_pos ? (end_pos - i) : MVL;
        if (!bit_flag)
        {
            __vr y_vblock = _vel_vld_vssl(8, &y[i], VL);
            _vel_vst_vssl(y_vblock, 8, &x[i], VL);
        }
        else
        {
            __vr y_vblock = _vel_vldlzx_vssl(4, &mask_y[i], VL);
            _vel_vstlot_vssl(y_vblock, 4, (uint64_t)&mask_x[i], VL);
        }
    }
}

void merge_res(elem_t *bfs_y, int *colidx_y, int start_pos, int nnz,
               elem_t *bfs_y_tmp, int *colidx_y_tmp, const int max_lanes, bool bit_flag)
{
    for (int i = 0; i < nnz; i += max_lanes)
    {
        int VL = (i + max_lanes) > nnz ? (nnz - i) : max_lanes;
        if (!bit_flag)
        {
            __vr y_vblock = _vel_vld_vssl(8, &bfs_y_tmp[i], VL);
            _vel_vst_vssl(y_vblock, 8, &bfs_y[start_pos + i], VL);
        }
        __vr colidx_y_vblock = _vel_vldlzx_vssl(4, &colidx_y_tmp[i], VL);
        _vel_vstl_vssl(colidx_y_vblock, 4, &colidx_y[start_pos + i], VL);
    }
}

void sparse2dense(elem_t *bfs_y, int *colidx_y, int nnz,
                  elem_t *bfs_x, int *mask_x,
                  int MVL, bool bit_flag, int nrows)
{
    for (int i = 0; i < nnz; i += MVL)
    {
        int VL = (i + MVL) > nnz ? (nnz - i) : MVL;
        __vr colidx_y_vblock = _vel_vldlzx_vssl(4, &colidx_y[i], VL);
        if (!bit_flag)
        {
            // scatter sparse y to dense x
            __vr vy_addr = _vel_vsfa_vvssl(colidx_y_vblock, 3UL, (unsigned long)bfs_x, VL);
            __vr y_vblock = _vel_vld_vssl(8, &bfs_y[i], VL);
            _vel_vsc_vvssl(y_vblock, vy_addr, (uint64_t)&bfs_x[0], (uint64_t)&bfs_x[nrows], VL);
        }
        else
        {
            __vr vy_addr = _vel_vsfa_vvssl(colidx_y_vblock, 2UL, (unsigned long)mask_x, VL);
            __vr vtmp = _vel_vbrdl_vsl(1UL, VL);
            _vel_vscl_vvssl(vtmp, vy_addr, (uint64_t)&mask_x[0], (uint64_t)&mask_x[nrows], VL);
        }
    }
}
void dense2sparse(elem_t *bfs_y, int *mask_y, int start_pos, int end_pos, int nnz_start,
                  elem_t *bfs_x, int *mask_x, int *colidx_x,
                  int MVL, bool bit_flag, int nrows)
{
    int nnz_tmp = 0;
    __vr vseq = _vel_vseq_vl(MVL);
    int nnz_mask = 0;
    for (int i = start_pos; i < end_pos; i += MVL)
    {
        int VL = (i + MVL) > end_pos ? (end_pos - i) : MVL;
        __vr vidx = _vel_vaddul_vsvl((unsigned long)i, vseq, VL);
        // __vr y_sc_addr = _vel_vldlzx_vssl(4, &row_order[i], VL);
        if (!bit_flag)
        {
            __vr y_vblock = _vel_vld_vssl(8, &bfs_y[i], VL);
            __vm256 vmask = _vel_vfmklgt_mvl(y_vblock, VL);
            unsigned nnz = _vel_pcvm_sml(vmask, VL);
            y_vblock = _vel_vcp_vvmvl(y_vblock, vmask, y_vblock, VL);
            vidx = _vel_vcp_vvmvl(vidx, vmask, vidx, VL);
            _vel_vst_vssl(y_vblock, 8, &bfs_x[nnz_start + nnz_tmp], nnz);
            _vel_vstl_vssl(vidx, 4, &colidx_x[nnz_start + nnz_tmp], nnz);
            nnz_tmp += nnz;
        }
        else
        {
            __vr masky_vblock = _vel_vldlzx_vssl(4, &mask_y[i], VL);
            __vm256 vmask = _vel_vfmklgt_mvl(masky_vblock, VL);
            unsigned nnz = _vel_pcvm_sml(vmask, VL);
            __vr colidx_vcp = _vel_vcp_vvmvl(vidx, vmask, vidx, VL);
            _vel_vstl_vssl(colidx_vcp, 4, &colidx_x[nnz_start + nnz_tmp], nnz);
            nnz_tmp += nnz;
        }
    }
}

void reset(int *arr, int start, int end, int MVL)
{
    for (int i = start; i < end; i += MVL)
    {
        int VL = i + MVL > end ? end - i : MVL;
        __vr varr = _vel_vbrdw_vsl(0, VL);
        _vel_vstl_vssl(varr, 4, &arr[i], VL);
    }
}

void bfs_kernel_spmspv_SELLCS_DFC_bypass(const SparseMatrixSELLCS *restrict matrix,
                                         const int *slice_idx_max,
                                         const int *slice_idx_min,
                                         const int *slice_nbypass,
                                         const int slice_num,
                                         const int *bucket,
                                         const elem_t *restrict x,
                                         const int *colidx_x,
                                         const int *nnzx,
                                         const int x_max,
                                         const int x_min,
                                         elem_t *restrict y,
                                         int *colidx_y,
                                         int *nnzy,
                                         const int start_slice,
                                         const int end_slice)
{
    const uint32_t vlen = matrix->C;
    int nrows = matrix->nrows;
    int nnz_tmp = 0;

    for (int sid = 0; sid < slice_num; sid++)
    {
        int slice_idx = slice_nbypass[sid];
        int row_idx = slice_idx << 8;
        uint32_t max_lanes = ((row_idx + vlen) > matrix->nrows) ? (matrix->nrows - row_idx) : vlen;
        int elem_idx = matrix->slice_pointers[slice_idx];
        elem_t *values_pointer = &matrix->values[elem_idx];
        uint64_t *colidx_pointer = &matrix->column_indices[elem_idx];

        __vr tmp_results = _vel_vxor_vvvl(tmp_results, tmp_results, max_lanes);
        int swidth = matrix->slice_widths[slice_idx];
        int act_lanes_idx = matrix->vop_pointers[slice_idx];
        __vr y_sc_addr = _vel_vldlzx_vssl(4, &matrix->row_order[row_idx], max_lanes);

        __vm256 result_mask = _vel_vfmklaf_ml(max_lanes);

        for (int i = 0; i < swidth; i++)
        {
            uint32_t act_lanes = (uint32_t)matrix->vop_lengths[act_lanes_idx++] + 1;
            // Load Column indices
            __vr col_index_vblock = _vel_vld_vssl(8, colidx_pointer, act_lanes);
            // Gather bucket
            __vr bucket_addr = _vel_vsfa_vvssl(col_index_vblock, 3UL, (unsigned long)bucket, act_lanes);
            __vr bucket_vblock = _vel_vgt_vvssl(bucket_addr, (uint64_t)&bucket[0], (uint64_t)bucket[nrows] + 1, act_lanes);
            // gather Colidx_x
            __vr colidx_x_vblock = _vel_vsfa_vvssl(bucket_vblock, 3UL, (unsigned long)colidx_x, act_lanes);
            colidx_x_vblock = _vel_vgt_vvssl(colidx_x_vblock, (uint64_t)&colidx_x[0], (uint64_t)colidx_x[(*nnzx)] + 1, act_lanes);
            __vr sub_vblock = _vel_vsubul_vvvl(colidx_x_vblock, col_index_vblock, act_lanes);
            // compare and get the mask
            __vm256 column_mask = _vel_vfmkleq_mvl(sub_vblock, act_lanes);
            __vm256 mask2 = _vel_vfmklaf_ml(256);
            unsigned long int match_cnt = _vel_pcvm_sml(column_mask, act_lanes);

            if (match_cnt != 0)
            {
                // Load Values
                __vr values_vblock = _vel_vld_vssl(8, values_pointer, act_lanes);
                // Gather X
                __vr x_vblock = _vel_vsfa_vvssl(bucket_vblock, 3UL, (unsigned long)colidx_x, act_lanes);
                x_vblock = _vel_vgt_vvssl(x_vblock, (uint64_t)&x[0], (uint64_t)x[(*nnzx)] + 1, act_lanes);

                // Multiply
                tmp_results = _vel_vfmadd_vvvvmvl(tmp_results, x_vblock, values_vblock, column_mask, tmp_results, act_lanes);
            }
            values_pointer += act_lanes;
            colidx_pointer += act_lanes;
            result_mask = _vel_orm_mmm(result_mask, column_mask);
        }
        unsigned long int slice_nnz = _vel_pcvm_sml(result_mask, max_lanes);
        // printf("slice nnz = %lu\n", slice_nnz);

        if (slice_nnz != 0)
        {
            __vr result_vcp = _vel_vcp_vvmvl(tmp_results, result_mask, tmp_results, max_lanes);
            __vr colidxy_vcp = _vel_vcp_vvmvl(y_sc_addr, result_mask, y_sc_addr, max_lanes);
            _vel_vst_vssl(result_vcp, 8, &y[nnz_tmp], slice_nnz);
            _vel_vst_vssl(colidxy_vcp, 8, &colidx_y[nnz_tmp], slice_nnz);
            nnz_tmp += slice_nnz;
        }
    }
    (*nnzy) = nnz_tmp;
}
