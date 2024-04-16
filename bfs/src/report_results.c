
#include "report_results.h"
#include <stdio.h>
#include <time.h>
#include "bfs.h"

#if defined(USE_OMP)
#include <omp.h> // enables use of get_num_cpus
#endif


void BFS_ReportResults(const SparseMatrixCSR *matrix, 
                       double conv_time,
                       double *iter_times,
                       double bfs_time,
                       int *frontier,
                       int num_iter, 
                       int num_vex,
                       bool bit_flag,
                       bool sort_flag)
{
    if (perf_analysis){
        printf("    \"Problem summary\": {\n");
        printf("        \"Input name\": \"%s\",\n", matrix->name);
        printf("        \"Matrix Num. Rows\": %i,\n", matrix->nrows);
        printf("        \"Matrix Num. Columns\": %i,\n", matrix->ncolumns);
        printf("        \"Total non-zero elements\": %i,\n", matrix->nnz);
        printf("        \"Non-zero elements per row\": %i,\n", matrix->nnz / matrix->nrows);
        printf("        \"Num. vertices \": %d\n", num_vex);
        printf("    },\n");

        printf("    \"Performance statistics\": { \n");
        printf("        \"Time converting to %s format [ms]\": %f,\n", algorithm_version, conv_time);
        printf("        \"%s BFS execution iterations \": %i,\n", algorithm_version, num_iter);
        printf("        \"%s BFS execution time [ms]\": %f,\n", algorithm_version, bfs_time);
        printf("        \"%s BFS TEPS\": %.2f\n", algorithm_version, matrix->nnz * 1.0 / (bfs_time * 1000000));
        printf("    },\n");
    }

    

    const char *last_slash_ptr = strrchr(matrix->name, '/'); // Find the last occurrence of '/'
    // Calculate the length of the substring
    int substring_length = strlen(last_slash_ptr + 1);
    // Allocate memory for the character array
    char char_array[substring_length + 1];
    // Copy the substring after the last '/' into the character array
    strcpy(char_array, last_slash_ptr + 1);
    char alg_filename[50];
    sprintf(alg_filename, "%s%s%s", "../bfs/data/", algorithm_version, "_runtime-results.csv");

    if (bypass_record == 0){
    FILE *filename = fopen(alg_filename, "a");
    fprintf(filename, "%s,%i,%i,%i,%i,%.3f", 
                      char_array, matrix->nrows, matrix->ncolumns, matrix->nnz, 
                      num_iter, bfs_time);

    if (perf_analysis)
    {
        for (int i = 0; i < num_iter; i ++)
        {
                fprintf(filename, ",%.4f", iter_times[i]);
        }
        fprintf(filename, "\n");
    }
    else{
        fprintf(filename, "\n");
    }
    fclose(filename);
    }

}

void BFS_verifyReports(const SparseMatrixCSR *matrix, 
                        int num_iter,
                        int *frontier)
{
    const char *last_slash_ptr = strrchr(matrix->name, '/'); // Find the last occurrence of '/'
    
    // Calculate the length of the substring
    int substring_length = strlen(last_slash_ptr + 1);   
    char char_array[substring_length + 1];
    strcpy(char_array, last_slash_ptr + 1);

    char alg_filename[50];
    sprintf(alg_filename, "%s%s%s", "data/", algorithm_version, "-nec-frontier.csv");

    FILE *filename = fopen(alg_filename, "a");
    fprintf(filename, "%s,%i,%i,%i,%i", 
                      char_array, matrix->nrows, matrix->ncolumns, matrix->nnz, 
                      num_iter);

    for (int i = 0; i < num_iter; i ++)
    {
        if (i < 200){
            fprintf(filename, ",%i", frontier[i]);
        }
    }
    fprintf(filename, "\n");
    fclose(filename);

}
