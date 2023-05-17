#ifndef NODECHIP_ABSADD_H_
#define NODECHIP_ABSADD_H_

#include "tpu_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_swapchannel_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    const int *order,
    data_type_t dtype);

#ifdef __cplusplus
}
#endif

#endif
