#ifndef NODECHIP_ABSADD_H_
#define NODECHIP_ABSADD_H_

#include "tpu_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_absadd_f32_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    float b_val,
    data_type_t dtype);

#ifdef __cplusplus
}
#endif

#endif
