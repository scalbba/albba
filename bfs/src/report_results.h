
#ifndef _REPORT_RESULTS_H_
#define _REPORT_RESULTS_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "sparse_matrix.h"

extern char algorithm_version[30];
extern size_t align_size;
extern int perf_analysis;
extern int verify;
extern int bypass_record;

void ReportResults(const SparseMatrixCSR *matrix, int verification, double *elapsed_times,
                   int num_iterations);

void BFS_ReportResults(const SparseMatrixCSR *matrix, 
                       double conv_time,
                       double *iter_times,
                       double bfs_time,
                       int *frontier,
                       int num_iter, 
                       int num_vex,
                       bool bit_flag,
                       bool sort_flag);

void BFS_verifyReports(const SparseMatrixCSR *matrix, 
                  int num_iter,
                  int *frontier_ref);


#endif /* _REPORT_RESULTS_H_ */
