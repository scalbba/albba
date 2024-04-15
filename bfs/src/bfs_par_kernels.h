/* Copyright 2020 Barcelona Supercomputing Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _SPMV_PAR_KERNELS_H_
#define _SPMV_PAR_KERNELS_H_
#include "common.h"
#include "sparse_matrix.h"

void kernel_SELLCS_FG(const SparseMatrixELLPACK *restrict matrix,
                      const elem_t *restrict x,
                      elem_t *restrict y,
                      const int start_row,
                      const int end_row,
                      const uint8_t *restrict vactive_lanes,
                      const int *restrict actlanes_ptr,
                      const int *restrict vrow_order,
                      const int *restrict slices_width,
                      const int *restrict slices_ptr);

void kernel_SELLCS_DFC(const SparseMatrixSELLCS *restrict matrix,
                       const elem_t *restrict x,
                       elem_t *restrict y,
                       const int start_row,
                       const int end_row);
void kernel_spmv_SELLCS_U8_NC_DFC(const SparseMatrixSELLCS *restrict matrix,
                                  const elem_t *restrict x,
                                  elem_t *restrict y,
                                  int *nnzy,
                                  const int start_slice,
                                  const int end_slice,
                                  size_t *mask);

void kernel_SELLCS_U8_NC_DFC_bitmap(const SparseMatrixSELLCS *restrict matrix,
                                    int *mask_x,
                                    int *mask_y,
                                    int *nnzy,
                                    const int start_slice,
                                    const int end_slice,
                                    size_t *check_mask);

void bfs_kernel_spmv_SELLCS_NC_DFC(const SparseMatrixSELLCS *restrict matrix,
                                   const elem_t *x,
                                   elem_t *y,
                                   int *nnzy,
                                   const int start_slice,
                                   const int end_slice,
                                   size_t *mask);

void bfs_kernel_SELLCS_DFC_bitmap(const SparseMatrixSELLCS *restrict matrix,
                                  const int *mask_x,
                                  int *mask_y,
                                  int *nnzy,
                                  const int start_slice,
                                  const int end_slice,
                                  size_t *mask,
                                  int *ypass_slice);

void bfs_kernel_SELLCS_DFC_bitmap_v2(const SparseMatrixSELLCS *restrict matrix,
                                     const int *mask_x,
                                     int *mask_y,
                                     int *nnzy,
                                     const int start_slice,
                                     const int end_slice,
                                     size_t *mask,
                                     int *compressed_x);

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
                                  int *bypass_width);
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
                                         int *bypass_width);

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
                                         const int end_slice);
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
                  size_t *mask);

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
                      size_t *mask);

void merge_res(elem_t *bfs_y, int *colidx_y, int start_pos, int nnz,
               elem_t *bfs_y_tmp, int *colidx_y_tmp, const int max_lanes, bool bit_flag);
void spmv_merge_res(elem_t *x, int *mask_x, int start_pos, int end_pos,
                    elem_t *y, int *mask_y, int MVL, bool bit_flag);
void spmspv_write_back_x(elem_t *x, int *colidx_x, int *nnzx,
                         elem_t *y, int *colidx_y, int nnzy,
                         int MVL, bool bit_flag);
void spmv_write_back_x(elem_t *x, int *mask_x, int *nnzx,
                       elem_t *y, int *mask_y, int nnzy,
                       int nrows, int MVL, bool bit_flag);
void sparse2dense(elem_t *bfs_y, int *colidx_y, int nnz,
                  elem_t *bfs_x, int *mask_x,
                  int MVL, bool bit_flag, int nrows);
void dense2sparse(elem_t *bfs_y, int *mask_y, int start_pos, int end_pos, int nnz_start,
                  elem_t *bfs_x, int *mask_x, int *colidx_x,
                  int MVL, bool bit_flag, int nrows);

void reset(int *arr, int start, int end, int MVL);

#endif
