#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"

// type infer function (default, no need to override)

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

int api_absadd_local_bfsz(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *input_spec,
    local_tensor_spec_t *output_spec,
    const void* param)
{
    int shape[4];
    parse_input_slice_shape(sec_info, input_spec, shape);
    return get_absadd_local_bfsz(
            shape,
            tpu_type_convert(input_spec->dtype));
}

// local api function
void api_absadd_local(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *input,
    local_tensor_spec_t *output,
    const void *param) {
    int shape[4];
    parse_input_slice_shape(sec_info, input, shape);
    PARSE_PARAM(absadd, absadd_param, param);
    tpu_impl_absadd_local(
        input->addr,
        output->addr,
        get_local_buffer_addr(param),
        shape,
        absadd_param.b_val,
        tpu_type_convert(input->dtype));
}
