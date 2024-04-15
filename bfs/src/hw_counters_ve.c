#include "common.h"
#include "hw_counters_ve.h"
#define __STDC_FORMAT_MACROS

const uint32_t kMaxLabelSize = 30;
const uint32_t kNumPMC = 16;

uint64_t *ve_init_hw_counters()
{
    return (uint64_t *)malloc(kNumPMC * sizeof(uint64_t));
}

void ve_hw_counters_start(uint64_t *restrict events)
{
    uint64_t pmc00, pmc01, pmc02, pmc03, pmc04, pmc05, pmc06, pmc07, pmc08,
        pmc09, pmc10, pmc11, pmc12, pmc13, pmc14, pmc15, usrcc;

    __asm__ volatile("smir %0, %%pmc00\n\t"
                     "smir %1, %%pmc01\n\t"
                     "smir %2, %%pmc02\n\t"
                     "smir %3, %%pmc03\n\t"
                     "smir %4, %%pmc04\n\t"
                     "smir %5, %%pmc05\n\t"
                     "smir %6, %%pmc06\n\t"
                     "smir %7, %%pmc07\n\t"
                     "smir %8, %%pmc08\n\t"
                     "smir %9, %%pmc09\n\t"
                     "smir %10, %%pmc10\n\t"
                     "smir %11, %%pmc11\n\t"
                     "smir %12, %%pmc12\n\t"
                     "smir %13, %%pmc13\n\t"
                     "smir %14, %%pmc14\n\t"
                     "smir %15, %%pmc15\n\t"
                     "smir %16, %%usrcc\n\t"
                     : "=r"(pmc00),
                       "=r"(pmc01),
                       "=r"(pmc02),
                       "=r"(pmc03),
                       "=r"(pmc04),
                       "=r"(pmc05),
                       "=r"(pmc06),
                       "=r"(pmc07),
                       "=r"(pmc08),
                       "=r"(pmc09),
                       "=r"(pmc10),
                       "=r"(pmc11),
                       "=r"(pmc12),
                       "=r"(pmc13),
                       "=r"(pmc14),
                       "=r"(pmc15),
                       "=r"(usrcc));

    events[0] = pmc00;
    events[1] = pmc01;
    events[2] = pmc02;
    events[3] = pmc03;
    events[4] = pmc04;
    events[5] = pmc05;
    events[6] = pmc06;
    events[7] = pmc07;
    events[8] = pmc08;
    events[9] = pmc09;
    events[10] = pmc10;
    events[11] = pmc11;
    events[12] = pmc12;
    events[13] = pmc13;
    events[14] = pmc14;
    events[15] = pmc15;
    events[16] = usrcc;
}

void ve_hw_counters_stop(uint64_t *restrict events)
{
    uint64_t *tmp = malloc(kNumPMC * sizeof(uint64_t));

    uint64_t pmc00, pmc01, pmc02, pmc03, pmc04, pmc05, pmc06,
        pmc07, pmc08, pmc09, pmc10, pmc11, pmc12, pmc13, pmc14, pmc15, usrcc;


    __asm__ volatile("smir %0, %%pmc00\n\t"
                     "smir %1, %%pmc01\n\t"
                     "smir %2, %%pmc02\n\t"
                     "smir %3, %%pmc03\n\t"
                     "smir %4, %%pmc04\n\t"
                     "smir %5, %%pmc05\n\t"
                     "smir %6, %%pmc06\n\t"
                     "smir %7, %%pmc07\n\t"
                     "smir %8, %%pmc08\n\t"
                     "smir %9, %%pmc09\n\t"
                     "smir %10, %%pmc10\n\t"
                     "smir %11, %%pmc11\n\t"
                     "smir %12, %%pmc12\n\t"
                     "smir %13, %%pmc13\n\t"
                     "smir %14, %%pmc14\n\t"
                     "smir %15, %%pmc15\n\t"
                     "smir %16, %%usrcc\n\t"
                     : "=r"(pmc00),
                       "=r"(pmc01),
                       "=r"(pmc02),
                       "=r"(pmc03),
                       "=r"(pmc04),
                       "=r"(pmc05),
                       "=r"(pmc06),
                       "=r"(pmc07),
                       "=r"(pmc08),
                       "=r"(pmc09),
                       "=r"(pmc10),
                       "=r"(pmc11),
                       "=r"(pmc12),
                       "=r"(pmc13),
                       "=r"(pmc14),
                       "=r"(pmc15),
                       "=r"(usrcc));

    tmp[0] = pmc00;
    tmp[1] = pmc01;
    tmp[2] = pmc02;
    tmp[3] = pmc03;
    tmp[4] = pmc04;
    tmp[5] = pmc05;
    tmp[6] = pmc06;
    tmp[7] = pmc07;
    tmp[8] = pmc08;
    tmp[9] = pmc09;
    tmp[10] = pmc10;
    tmp[11] = pmc11;
    tmp[12] = pmc12;
    tmp[13] = pmc13;
    tmp[14] = pmc14;
    tmp[15] = pmc15;
    tmp[16] = usrcc;

    events[0] = tmp[0] - events[0];
    events[1] = tmp[1] - events[1];
    events[2] = tmp[2] - events[2];
    events[3] = tmp[3] - events[3];
    events[4] = tmp[4] - events[4];
    events[5] = tmp[5] - events[5];
    events[6] = tmp[6] - events[6];
    events[7] = tmp[7] - events[7];
    events[8] = tmp[8] - events[8];
    events[9] = tmp[9] - events[9];
    events[10] = tmp[10] - events[10];
    events[11] = tmp[11] - events[11];
    events[12] = tmp[12] - events[12];
    events[13] = tmp[13] - events[13];
    events[14] = tmp[14] - events[14];
    events[15] = tmp[15] - events[15];
    events[16] = tmp[16] - events[16];
}

void ve_print_hardware_counters(const char *section, const uint64_t *ev)
{
    const char *mode = getenv("VE_PERF_MODE");


    if (strcmp(mode,"VECTOR-OP") == 0)
    {
        fprintf(stderr, "{\n");
        fprintf(stderr, "    \"VE counters\": { \n");
        fprintf(stderr, "        \"Section\": \"%s\",\n", section);
        fprintf(stderr, "        \"Exec. Inst.\": %" PRIu64 ",\n", ev[0]);
        fprintf(stderr, "        \"Vec. Inst. Execution\": %" PRIu64 ",\n", ev[1]);
        fprintf(stderr, "        \"FP Data Elements\": %" PRIu64 ",\n", ev[2]);
        fprintf(stderr, "        \"Vec. Elements\": %" PRIu64 ",\n", ev[3]);
        fprintf(stderr, "        \"Vec. Exec. Cycles\": %" PRIu64 ",\n", ev[4]);
        fprintf(stderr, "        \"L1$ Miss Cycles\": %" PRIu64 ",\n", ev[5]);
        fprintf(stderr, "        \"Vec. Elements 2\": %" PRIu64 ",\n", ev[6]);
        fprintf(stderr, "        \"Vec. Arithmetic exec. Cycles\": %" PRIu64 ",\n", ev[7]);
        fprintf(stderr, "        \"Vec. Load execution Cycles\": %" PRIu64 ",\n", ev[8]);
        fprintf(stderr, "        \"Port Conflict Cycles\": %" PRIu64 ",\n", ev[9]);
        fprintf(stderr, "        \"Vec. Loaded Packets\": %" PRIu64 ",\n", ev[10]);
        fprintf(stderr, "        \"Vec. Loaded elements\": %" PRIu64 ",\n", ev[11]);
        fprintf(stderr, "        \"Vec. Load Cache Miss Elements\": %" PRIu64 ",\n", ev[12]);
        fprintf(stderr, "        \"FMA Element Count\": %"PRIu64",\n", ev[13]);
        // fprintf(stderr, "        \"Power Throttling Cycles\": %"PRIu64",\n", ev[14]);
        // fprintf(stderr, "        \"Thermal Throttling Cycles\": %"PRIu64",\n", ev[15]);
        fprintf(stderr, "        \"User Cycles\": %" PRIu64 "\n", ev[16]);
        fprintf(stderr, "    }\n");
        fprintf(stderr, "}\n");
    }
    else if (strcmp(mode,"VECTOR-MEM") == 0)
    {
        fprintf(stderr, "{\n");
        fprintf(stderr, "    \"VE counters\": { \n");
        fprintf(stderr, "        \"Section\": \"%s\",\n", section);
        fprintf(stderr, "        \"Exec. Inst.\": %" PRIu64 ",\n", ev[0]);
        fprintf(stderr, "        \"Vec. Inst. Execution\": %" PRIu64 ",\n", ev[1]);
        fprintf(stderr, "        \"FP Data Elements\": %" PRIu64 ",\n", ev[2]);
        fprintf(stderr, "        \"L1I$ Miss Count\": %" PRIu64 ",\n", ev[3]);
        fprintf(stderr, "        \"L1I$ Access Count\": %" PRIu64 ",\n", ev[4]);
        fprintf(stderr, "        \"L1D$ Miss Count\": %" PRIu64 ",\n", ev[5]);
        fprintf(stderr, "        \"L1D$ Access Count\": %" PRIu64 ",\n", ev[6]);
        fprintf(stderr, "        \"L2$ Miss Count\": %" PRIu64 ",\n", ev[7]);
        fprintf(stderr, "        \"L2$ Access Count\": %" PRIu64 ",\n", ev[8]);
        fprintf(stderr, "        \"Branch Exec. Count\": %" PRIu64 ",\n", ev[9]);
        fprintf(stderr, "        \"Branch Pred. Fail Count\": %" PRIu64 ",\n", ev[10]);
        fprintf(stderr, "        \"Vec. Load Exec. Count\": %" PRIu64 ",\n", ev[11]);
        fprintf(stderr, "        \"Vec. Load Miss Exec. Count\": %" PRIu64 ",\n", ev[12]);
        fprintf(stderr, "        \"FMA Exec. Count\": %"PRIu64",\n", ev[13]);
        //fprintf(stderr, "        \"Power Throttling Cycles\": %"PRIu64",\n", ev[14]);
        // fprintf(stderr, "        \"Thermal Throttling Cycles\": %"PRIu64",\n", ev[15]);
        fprintf(stderr, "        \"User Cycles\": %" PRIu64 "\n", ev[16]);
        fprintf(stderr, "    }\n");
        fprintf(stderr, "}\n");
    }
    else{
        fprintf(stderr, "{\n");
        fprintf(stderr, "    \"VE counters\": { \n");
        fprintf(stderr, "        \"Section\": \"%s\",\n", section);
        fprintf(stderr, "        \"LOAD TRAFFIC Count\": %" PRIu64 ",\n", ev[9]);
        fprintf(stderr, "        \"STORE TRAFFIC Count\": %" PRIu64 ",\n", ev[10]);
        fprintf(stderr, "        \"User Cycles\": %" PRIu64 "\n", ev[16]);
        fprintf(stderr, "    }\n");
        fprintf(stderr, "}\n");
    }
}
