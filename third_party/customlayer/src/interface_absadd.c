#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"

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
    PARSE_PARAM(absadd, absadd_param, param);
    tpu_impl_absadd_global(
        input->addr,
        output->addr,
        input->shape,
        absadd_param.b_val,
        tpu_type_convert(input->dtype));
}

// local api function
