#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <sys/time.h>
#include <libgen.h>
#include <getopt.h> // Required if using -std=cXX
#include <unistd.h>

#ifndef SPMV
#define SPMV 0
#endif

#ifndef SORT_BY_COLIDX
#define SORT_BY_COLIDX 1
#endif

#ifndef BITMAP
#define BITMAP 1
#endif

#ifndef DIRECTED
#define DIRECTED 1
#endif

#endif
