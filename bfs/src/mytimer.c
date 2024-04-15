
#include <stddef.h>
#include <sys/resource.h>
#include "mytimer.h"
#include <sys/time.h>

double mytimer(void)
{
    // gettimeof day is precise enough in Linux
    // https://software.intel.com/en-us/itc-user-and-reference-guide-posix-clock-gettime

    struct timeval tp;
    static long start = 0, startu;
    if (!start)
    {
        gettimeofday(&tp, NULL);
        start = tp.tv_sec;
        startu = tp.tv_usec;
        return 0.0;
    }
    gettimeofday(&tp, NULL);
    return ((double)(tp.tv_sec - start)) + (tp.tv_usec - startu) / 1000000.0;
}
