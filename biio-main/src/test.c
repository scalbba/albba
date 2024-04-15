#include "biio.h"

int main(int *argc, char *argv[])
{
    int row;
    int col;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *val;
    int isSymmeticeR;
    char *filename = "MOLIERE_2016.cbd";
    read_Dmatrix(&row, &col, &nnz, &row_ptr, &col_idx, &val, &isSymmeticeR, filename);
    INFO_LOG("read_Dmatrix, %s\n", filename);
    INFO_LOG("row: %d\n", row);
    INFO_LOG("col: %d\n", col);
    INFO_LOG("nnz: %d\n", nnz);
    // should be 6,669,254,694
    return 0;
}