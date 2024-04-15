
#ifndef _MACROS_H_
#define _MACROS_H_

#ifdef ENABLE_HWC_VE // If ENABLE_HWC_VE is defined PMC will be used, otherwise wont have any effect.
    #include "hw_counters_ve.h"
  #define HW_COUNTERS(x) x 
#else
  #define HW_COUNTERS(x)
#endif

#ifdef DEBUG
  #define DEBUG_MSG(x) x
#else
  #define DEBUG_MSG(x)
#endif


#endif
