#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"

void api_preprocess_global(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    PARSE_PARAM(preprocess, preprocess_param, param);
    tpu_impl_preprocess_global(
        input->addr,
        output->addr,
        input->shape,
        preprocess_param.scale,
        preprocess_param.mean,
        tpu_type_convert(input->dtype),
        tpu_type_convert(output->dtype));
}