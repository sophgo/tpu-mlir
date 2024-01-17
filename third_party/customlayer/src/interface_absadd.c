#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "backend_custom_param.h"

// parse param function
static absadd_param_t absadd_parse_param(const void* param) {
    absadd_param_t abs_param = {0};
    abs_param.b_val = ((custom_param_t *)param)[0].float_t;
    return abs_param;
}

// shape infer function
void shape_infer_absadd(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    output->dtype = input->dtype;
    output->dims = input->dims;
    memcpy(output->shape, input->shape, output->dims);
    output->elem_num = input->elem_num;
}

// global api function
void api_absadd_global(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    absadd_param_t absadd_param = absadd_parse_param(param);
    tpu_impl_absadd_global(
        input->addr,
        output->addr,
        input->shape,
        absadd_param.b_val,
        tpu_type_convert(input->dtype));
}

// local api function
