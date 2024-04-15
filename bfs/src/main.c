#include "common.h"
#include "sparse_matrix.h"
#include "bfs.h"
#include "utils.h"
#include "report_results.h"
#include "mytimer.h"
#include "biio.h"
#include "bfs_ref.h"

extern enum MatrixFormat required_matrix_format;
extern int sigma_window;
extern int num_blocks;
extern int chunk_size;
int perf_analysis = 0;
int verify = 0;
int bypass_record = 0;

#ifdef ALIGN_TO
size_t align_size = ALIGN_TO;
#else
size_t align_size = 64;
#endif

void print_usage()
{
    printf("Usage: ./spmv -m <mtx filepath>]\n");
}


int main(int argc, char **argv)
{
    char *mtx_filepath = "";
    int opt, verification;
    int num_iterations = 10;
    double elapsed_times[15];
    elem_t *y_ref, *y, *x;
    SparseMatrixCOO tmp_matrix;
    SparseMatrixCSR csr_matrix;
    bool bit_flag = 0;
    bool sort_flag= 0;



    while ((opt = getopt(argc, argv, ":i:m:w:b:c:s:v:p:br:")) != -1)
    {
        switch (opt)
        {
        case 'i':
            num_iterations = atoi(optarg);
            break;
        case 'm':
            mtx_filepath = optarg;
            break;
        case 'w':
            sigma_window = atoi(optarg);
            break;
        case 'b':
            num_blocks = atoi(optarg);
            break;
        case 'c':
            chunk_size = atoi(optarg);
            break;
        case 's':
            sort_flag = atoi(optarg);
            break;
        case 'v':
            verify = atoi(optarg);
            break;
        case 'p':
            perf_analysis = atoi(optarg);
            break;
        case 'br':
            bypass_record = atoi(optarg);
            break;
        case ':':
            printf("Missing value in option \'%c\'\n", optopt);
            print_usage();
            exit(1);
        case '?':
            printf("Unknown option \'%c\'\n", optopt);
            break;
        }
    }
    struct timeval t1, t2;

    //Loading matrix in CSR format
    gettimeofday(&t1, NULL);
    if (strcmp(mtx_filepath, ""))
    {
        csr_matrix.name = mtx_filepath;
        read_Dmatrix(&csr_matrix.nrows, &csr_matrix.ncolumns, &csr_matrix.nnz, 
                    &csr_matrix.row_pointers, &csr_matrix.column_indices, &csr_matrix.values, &csr_matrix.isSymmetric, csr_matrix.name);
    }
    else
    {
        printf("You must specify a valid input method\n");
        print_usage();
        exit(1);
    }
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", csr_matrix.nrows, csr_matrix.ncolumns, csr_matrix.nnz, time_loadmat/1000.0);

    
    //initialize the matrix
    for (int i = 0; i < csr_matrix.nnz; i ++)
    {
        csr_matrix.values[i] = 1.0;
    }

    /* Allocate and initialize the x vector */
    x = (elem_t *)aligned_alloc(align_size, csr_matrix.ncolumns * sizeof(elem_t));
    check_mem_alloc(x, "X vector");
    init_x(x, csr_matrix.ncolumns, 2);


  #if DIRECTED
    //directed graphs
        SparseMatrixCSC csc_matrix;
        convert_csr_to_csc(&csr_matrix, &csc_matrix);
        csr_matrix.row_pointers = csc_matrix.column_pointers;
        csr_matrix.column_indices = csc_matrix.row_indices;
        csr_matrix.values = csc_matrix.values;
        csr_matrix.nrows = csc_matrix.ncolumns;
        csr_matrix.ncolumns = csc_matrix.nrows;
        csr_matrix.nnz = csc_matrix.nnz;
  #else


    //convert to undireted graphs

   if(csr_matrix.isSymmetric==0)
    {
        SparseMatrixCSC csc_matrix;
        convert_csr_to_csc(&csr_matrix, &csc_matrix);
    
        int *csrRowPtr_AAT = (int *)aligned_alloc(align_size, (csr_matrix.ncolumns + 1) * sizeof(int));
        int *csrColIdx_AAT = (int *)aligned_alloc(align_size, csr_matrix.nnz * sizeof(int)*2); 
        int AATlen=0;
        for(int i=0; i<csr_matrix.nrows; i++)
        {
            int len1= csc_matrix.column_pointers[i+1] - csc_matrix.column_pointers[i];
            int len2=csr_matrix.row_pointers[i+1] - csr_matrix.row_pointers[i];
            csrRowPtr_AAT[i]=MergeArr(csc_matrix.row_indices + csc_matrix.column_pointers[i], len1, 
                                      csr_matrix.column_indices + csr_matrix.row_pointers[i],len2,
                                      csrColIdx_AAT + AATlen);
            AATlen+=csrRowPtr_AAT[i];
        }
        exclusive_scan(csrRowPtr_AAT,csr_matrix.ncolumns + 1);

        csr_matrix.nnz =0;

        for(int i=0;i<csr_matrix.nrows; i++)
        {
            csr_matrix.row_pointers[i]=0;
            for(int j=csrRowPtr_AAT[i]; j<csrRowPtr_AAT[i+1]; j++)
            {
                csr_matrix.row_pointers[i]++;
                csr_matrix.column_indices[csr_matrix.nnz++] = csrColIdx_AAT[j];          
            }
        }
   
        csr_matrix.row_pointers[csr_matrix.nrows]=0;
        exclusive_scan(csr_matrix.row_pointers, csr_matrix.nrows+1);
   
    
        free(csrRowPtr_AAT);
        free(csrColIdx_AAT);
    }
    
  #endif

    //initialize bfs_x 
    elem_t *bfs_x = (elem_t *)aligned_alloc(align_size, csr_matrix.nrows * sizeof(elem_t));
    int *colidx_x = (int *)aligned_alloc(align_size, csr_matrix.nrows * sizeof(int));
    int nnzx;

    int num_iter = 0;
    int num_vex = 1;
    int *frontier = (int *)aligned_alloc(align_size, csr_matrix.nrows * sizeof(int));
    double bfs_time =0;
    double *iter_times = (double *)aligned_alloc(align_size, csr_matrix.nrows * sizeof(double));
    double conv_time = 0;

    //Run BFS 
    

    init_bfs(bfs_x, colidx_x, &nnzx, &num_iter, &num_vex, frontier, &bfs_time, iter_times, &conv_time, csr_matrix.nrows);
    bit_flag = 1; 
    run_custom_bfs_test(&csr_matrix, bfs_x, colidx_x, &nnzx, frontier, &num_vex, &conv_time, iter_times, &bfs_time, &num_iter, bit_flag, sort_flag);

    if (verify){
        BFS_verifyReports(&csr_matrix, num_iter, frontier);
    

        // Run BFS reference
        int iter_ref = 0;
        int sum_vex_ref = 1;
        int *num_vex_ref = (int *)aligned_alloc(align_size, csr_matrix.nrows * sizeof(int));
        memset(bfs_x, 0, csr_matrix.nrows * sizeof(elem_t));
        memset(colidx_x, 0, csr_matrix.nrows * sizeof(int));
        nnzx = 1;
        bfs_x[0] = 1;
        colidx_x[0] = 0;

        run_bfs_ref(&csr_matrix, bfs_x, colidx_x, &nnzx,
                    &iter_ref, &sum_vex_ref, num_vex_ref);
        

        BFS_verifyReports(&csr_matrix, iter_ref, num_vex_ref);
    }

    BFS_ReportResults(&csr_matrix, conv_time,  iter_times, bfs_time, frontier, num_iter, num_vex,bit_flag, sort_flag);
    return 0;
}
