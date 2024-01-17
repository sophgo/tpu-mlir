#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "backend_custom_param.h"

// parse param function
static swapchannel_param_t swapchannel_parse_param(const void* param) {
    swapchannel_param_t sc_param = {0};
    for (int i = 0; i < 3; i++) {
        sc_param.order[i] = ((custom_param_t *)param)[0].int_arr_t[i];
    }
    return sc_param;
}

// shape infer function
void shape_infer_swapchannel(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    output->dtype = input->dtype;
    output->dims = input->dims;
    memcpy(output->shape, input->shape, output->dims);
    output->elem_num = input->elem_num;
}

// global api function
void api_swapchannel_global(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    swapchannel_param_t sc_param = swapchannel_parse_param(param);
    tpu_impl_swapchannel_global(
        input->addr,
        output->addr,
        input->shape,
        sc_param.order,
        tpu_type_convert(input->dtype));
}

// local api function
