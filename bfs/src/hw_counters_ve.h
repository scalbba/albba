

#ifndef _HW_COUNTERS_H_
#define _HW_COUNTERS_H_

#include "common.h"

typedef struct ve_pmc_t
{
    int num_of_events;
    char **labels;
    int *codes;
    long long *counter_values;
    double start_time_usec;
    double total_time_usec; 
} ve_pmc_t;

uint64_t * ve_init_hw_counters();
void ve_hw_counters_start(uint64_t *restrict events);
void ve_hw_counters_stop(uint64_t *restrict events);
void ve_print_hardware_counters(const char *section, const uint64_t *ev);


#endif /* _HW_COUNTERS_H_ */
