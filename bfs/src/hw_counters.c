#include "common.h"
#include "hw_counters.h"
#include "papi.h"


const uint32_t kMaxLabelSize = 30;
const uint32_t kMaxNumEvents = 20;

void init_hw_counters(papi_events_t *restrict ev)
{

    // ev = (papi_events_t *)malloc(sizeof(papi_events_t));
    ev->num_of_events = 0;
    ev->total_time_usec = 0;

    ev->codes = (int *) malloc(sizeof(int) * kMaxNumEvents);
    ev->labels = (char **) malloc(sizeof(char *) * kMaxNumEvents);
    for(int i = 0; i  < kMaxNumEvents; i++)
        ev->labels[i] = (char *) malloc(sizeof(char) * kMaxLabelSize);
        
    ev->counter_values = malloc(sizeof(long long) * kMaxNumEvents);
    
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    {
        fprintf(stderr, "PAPI couldn't init library properly \n");
        exit(1);
    }

    add_hw_counter(ev, "Cycles", PAPI_TOT_CYC);
    add_hw_counter(ev, "Instructions", PAPI_TOT_INS);
    add_hw_counter(ev, "Loads", PAPI_LD_INS);
    add_hw_counter(ev, "L1D$ Misses", PAPI_L1_DCM);
    add_hw_counter(ev, "L3 Accesses", PAPI_L3_TCA);
    add_hw_counter(ev, "L3 Misses", PAPI_L3_TCM);
    // add_hw_counter(ev, "DTLB Misses", PAPI_TLB_DM);
    add_hw_counter(ev, "Total Stall Cycles", PAPI_RES_STL);
    add_hw_counter(ev, "Mem. Write Stall Cycles", PAPI_MEM_WCY);

};

void add_hw_counter(papi_events_t *restrict ev, const char *restrict label, const int papi_code)
{
    if (strlen(label) >= kMaxLabelSize)
    {
        fprintf(stderr, "Error while adding hardware counters: label too long.\n");
    }
        
    strcpy(ev->labels[ev->num_of_events], label);
    ev->codes[ev->num_of_events] = papi_code;
    ev->num_of_events++;
}

void hw_counters_start(papi_events_t *restrict events)
{
    events->start_time_usec = PAPI_get_real_usec();
    
    handle_error(PAPI_start_counters(events->codes, events->num_of_events), "hw_counters_start");
}

void hw_counters_stop(papi_events_t *restrict events)
{
    // Time will be incorrectly measured if you do several start-stops
    events->total_time_usec += ((PAPI_get_real_usec() - events->start_time_usec) * 1e-6);
    handle_error(PAPI_stop_counters(events->counter_values, events->num_of_events), "hw_counters_stop");
      
}

void handle_error(int ret, const char *location)
{
    if (ret == PAPI_OK)
        return;
    fprintf(stderr, "PAPI Error: %s", PAPI_strerror(ret));
    fprintf(stderr, " (%d)\n", ret);
    fprintf(stderr, "  Occurred in %s\n", location);
    exit(1);
}

void print_hardware_counters(const char *section, const papi_events_t *ev)
{
    fprintf(stderr, "{\n");
    fprintf(stderr, "    \"PAPI counters\": { \n");
    fprintf(stderr, "        \"Section\": %s,\n", section);
    for (int e = 0; e < ev->num_of_events-1; e++)
    {
        fprintf(stderr, "        \"%s\": %lld,\n", ev->labels[e], ev->counter_values[e]);
    }
    fprintf(stderr, "        \"%s\": %lld\n", ev->labels[ev->num_of_events-1], ev->counter_values[ev->num_of_events-1]);
    fprintf(stderr, "    }\n");
    fprintf(stderr, "}\n");
}
