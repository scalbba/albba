
#ifndef _HW_COUNTERS_H_
#define _HW_COUNTERS_H_

#include "common.h"
extern char algorithm_version[30];

typedef struct papi_events_t
{
    int num_of_events;
    char **labels;
    int *codes;
    long long *counter_values;
    double start_time_usec;
    double total_time_usec; 
} papi_events_t;

void init_hw_counters(papi_events_t *ev);
void add_hw_counter(papi_events_t *ev, const char *label, const int papi_code);
void hw_counters_start(papi_events_t *events);
void hw_counters_stop(papi_events_t *events);
void handle_error(int ret, const char *location);
void print_hardware_counters(const char *section, const papi_events_t *ev);

#endif /* _HW_COUNTERS_H_ */
