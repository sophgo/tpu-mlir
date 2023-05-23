#include "tpu_utils.h"
#include "api_absadd.h"
#include "nodechip_absadd.h"

// parse param function
absadd_param_t absadd_parse_param(custom_param_t* param) {
    absadd_param_t abs_param = {0};
    abs_param.b_val = param[0].float_t;
    return abs_param;
}

// global api function
void api_absadd_global(
    global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    custom_param_t *param)
{
    absadd_param_t absadd_param = absadd_parse_param(param);

    nodechip_absadd_f32_global(
        input->addr,
        output->addr,
        input->shape,
        absadd_param.b_val,
        tpu_type_convert(input->dtype));
}

// local api function
