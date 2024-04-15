

#ifndef SORTING_UTILS_H
#define SORTING_UTILS_H

#include "sparse_matrix.h"
// extern int align_size;

int *get_order_by_row_size_radix(int *rows_size, const int nrows, int sigma_ordering_window);
void get_order_by_row_size_radix_sorted(int *row_order, int *rows_size, const int nrows, int sigma_ordering_window);
int *sort_by_1st_colidx(int *row_pointers, int *column_indices, int nrows);
void quickSort(int *a, int *b, int low, int high);



#endif // SORTING_UTILS_H
