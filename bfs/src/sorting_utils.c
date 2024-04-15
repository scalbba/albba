#ifndef SORTING_UTILS_C
#define SORTING_UTILS_C

#include "sorting_utils.h"
#include "utils.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>





/* Vector RADIX SORT */
uint64_t get_max_value(int* order_vector, int start, int end)
{
    uint64_t max_val = order_vector[start];
    for (int i = start + 1; i < end; i++)
        if (order_vector[i] > max_val)
            max_val = order_vector[i];

    return max_val;
}

const uint64_t NBINS = 16; // depends on base_p2
const uint64_t base_p2 = 4; // we will use 2^4 bins
const uint64_t base_mod_mask = 0xF; // depends on base_p2


void swap_key(int *a , int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_val(int *a , int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key-value pair (child function)
int partition_key_val_pair(int *key, int *val, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_val(&val[pivot_index], &val[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key(&key[pivot_index+i], &key[small_length]);
            swap_val(&val[pivot_index+i], &val[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);
    swap_val(&val[pivot_index + length - 1], &val[small_length]);

    return small_length;
}

// quick sort key-value pair (main function)
void quick_sort_key_val_pair(int *key, int *val, int length)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition_key_val_pair(key, val, length, 0) ;
    quick_sort_key_val_pair(key, val, small_length);
    quick_sort_key_val_pair(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
}

// quick sort key (child function)
int partition_key(int *key, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key(&key[pivot_index+i], &key[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);

    return small_length;
}

// quick sort key (main function)
void quick_sort_key(int *key, int length)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition_key(key, length, 0) ;
    quick_sort_key(key, small_length);
    quick_sort_key(&key[small_length + 1], length - small_length - 1);
}


void reverse_count_sort_p2(int* order_vector, int* paired_vector, int start, int end, uint64_t exp)
{
    const int len = end - start;
    int tmp_array[len];
    int tmp_array_paired[len];

    int* bins = (int*)malloc(NBINS * sizeof(int));
    memset(bins, 0, NBINS * sizeof(int));
    // Histogram of the values in NNZ
    for (int i = start; i < end; i++) {
        int bin_id = (order_vector[i] >> exp) & base_mod_mask;
        bins[bin_id]++;
    }
    // Indexes to insert values in reverse
    int cum_sum = 0;
    for (int i = 0; i < NBINS; i++) {
        cum_sum += bins[i];
        bins[i] = cum_sum;
    }
    for (int i = start; i < end; i++) {
        int bin_id = (order_vector[i] >> exp) & base_mod_mask;
        // Reverse the write index (position)
        int write_at_index = len - bins[bin_id];
        tmp_array[write_at_index] = order_vector[i];
        tmp_array_paired[write_at_index] = paired_vector[i];
        bins[bin_id]--;
    }
    memcpy(&order_vector[start], tmp_array, len * sizeof(int));
    memcpy(&paired_vector[start], tmp_array_paired, len * sizeof(int));

    free(bins);
}

void radix_sort_paired_descending(int* order_vector, int* paired_vector, int start, int end)
{
    // assert(end-start % 256);
    uint64_t maxvalue = get_max_value(order_vector, start, end);
    uint64_t max_num_digits = 0;
    while (maxvalue) {
        max_num_digits++;
        maxvalue = maxvalue >> base_p2;
    }

    int exp = 0;
    for (uint64_t i = 0; i < max_num_digits; i++) {
        reverse_count_sort_p2(order_vector, paired_vector, start, end, exp);
        exp += base_p2;
    }
}

//////





void reverse_count_sort_p2_opt(int* order_vector, int* paired_vector, int start, int end, uint64_t exp,
    int* bins, int* tmp_array, int* tmp_array_paired)
{
    const int len = end - start;
    // int tmp_array[len];
    // int tmp_array_paired[len];
    memset(bins, 0, NBINS * sizeof(int));

    // Histogram of the values in NNZ
    for (int i = start; i < end; i++) {
        int bin_id = (order_vector[i] >> exp) & base_mod_mask;
        bins[bin_id]++;
    }
    // Indexes to insert values in reverse
    int cum_sum = 0;
    for (int i = 0; i < NBINS; i++) {
        cum_sum += bins[i];
        bins[i] = cum_sum;
    }
    for (int i = start; i < end; i++) {
        const int bin_id = (order_vector[i] >> exp) & base_mod_mask;
        // Reverse the write index (position)
        const int write_at_index = len - bins[bin_id];
        tmp_array[write_at_index] = order_vector[i];
        tmp_array_paired[write_at_index] = paired_vector[i];
        bins[bin_id]--;
    }
    memcpy(&order_vector[start], tmp_array, len * sizeof(int));
    memcpy(&paired_vector[start], tmp_array_paired, len * sizeof(int));
}




void radix_sort_paired_descending_opt(int* order_vector, int* paired_vector, int start, int end)
{

    int* bins = (int*)malloc( NBINS * sizeof(int));

    // assert(end-start % 256);
    uint64_t maxvalue = get_max_value(order_vector, start, end);
    uint64_t max_num_digits = 0;
    while (maxvalue) {
        max_num_digits++;
        maxvalue = maxvalue >> base_p2;
    }
    int tmp_array[end - start];
    int tmp_array_paired[end - start];

    int exp = 0;
    for (uint64_t i = 0; i < max_num_digits; i++) {
        reverse_count_sort_p2_opt(order_vector, paired_vector, start, end, exp, bins, tmp_array, tmp_array_paired);
        exp += base_p2;
    }

    free(bins);

}

int* get_order_by_row_size_radix(int* rows_size, const int nrows, const int sigma_ordering_window)
{
    // Post: rows_size is ordered paired with row_order which contains the offset index (between 0 and sigma_ordering_window-1)
    //       corresponding to the order inside each 'ordering window'.

    int* row_order = (int*)aligned_alloc(align_size, nrows * sizeof(int));
    check_mem_alloc(row_order, "get_order_by_row_size.row_order\n");


    // Set a block of row_order between 0 and vlen-1
    for (int i = 0; i < nrows; i++)
        row_order[i] = i;

    #pragma omp parallel for schedule(dynamic,1)
    for (int k = 0; k < nrows; k += sigma_ordering_window) {
        int row_end = (k + sigma_ordering_window > nrows) ? nrows : k + sigma_ordering_window;
        // int sort_size = row_end - k;
        radix_sort_paired_descending_opt(&rows_size[0], &row_order[0], k, row_end);
    }

    return row_order;
}

int *sort_by_1st_colidx(int *row_pointers, int *column_indices, int nrows)
{
    int *row_order = (int*)aligned_alloc(align_size, nrows * sizeof(int));
    check_mem_alloc(row_order, "get_order_by_row_size.row_order\n");
    for (int i = 0; i < nrows; i++)
        row_order[i] = i;
    int *col_1st_tmp = (int*)aligned_alloc(align_size, nrows * sizeof(int));
    check_mem_alloc(col_1st_tmp, "column indices of the first elements of rows\n");
    memset(col_1st_tmp, 0, nrows * sizeof(int));
    for (int i = 0; i < nrows; i ++)
    {
        // if (row_pointers[i + 1] - row_pointers[i] != 0)
        // {
            int pos = row_pointers[i];
            col_1st_tmp[i] = column_indices[pos];
        // }
        // else{
        //     col_1st_tmp[i] = 0;
        // }
    }
    // quick_sort_key_val_pair(col_1st_tmp, row_order, nrows);
    radix_sort_paired_descending_opt(&col_1st_tmp[0], &row_order[0], 0, nrows);

    free(col_1st_tmp);

    return row_order;
}
void get_order_by_row_size_radix_sorted(int *row_order,
                                        int* rows_size, 
                                        const int nrows, 
                                        const int sigma_ordering_window)
{
    // Post: rows_size is ordered paired with row_order which contains the offset index (between 0 and sigma_ordering_window-1)
    //       corresponding to the order inside each 'ordering window'.


    #pragma omp parallel for schedule(dynamic,1)
    for (int k = 0; k < nrows; k += sigma_ordering_window) {
        int row_end = (k + sigma_ordering_window > nrows) ? nrows : k + sigma_ordering_window;
        // int sort_size = row_end - k;
        radix_sort_paired_descending_opt(&rows_size[0], &row_order[0], k, row_end);
    }
}


#endif
