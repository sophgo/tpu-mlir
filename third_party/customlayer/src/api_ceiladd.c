#include "tpu_utils.h"
#include "api_ceiladd.h"
#include "nodechip_ceiladd.h"

// parse param function
ceiladd_param_t ceiladd_parse_param(custom_param_t* param) {
    ceiladd_param_t ceiladd_param = {0};
    ceiladd_param.b_val = param[0].float_t;

    return ceiladd_param;
}

// global api function
void api_ceiladd_global(
    global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    custom_param_t *param)
{
    ceiladd_param_t ceiladd_param = ceiladd_parse_param(param);

    nodechip_ceiladd_f32_global(
        input->addr,
        output->addr,
        input->shape,
        ceiladd_param.b_val,
        tpu_type_convert(input->dtype));
}

// local api function
