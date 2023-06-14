#include "tpu_utils.h"
#include "api_swapchannel.h"
#include "nodechip_swapchannel.h"

// parse param function
swapchannel_param_t swapchannel_parse_param(custom_param_t* param) {
    swapchannel_param_t sc_param = {0};
    for (int i = 0; i < 3; i++) {
        sc_param.order[i] = param[0].int_arr_t[i];
    }
    return sc_param;
}

// global api function
void api_swapchannel_global(
    global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    custom_param_t *param)
{
    swapchannel_param_t sc_param = swapchannel_parse_param(param);

    nodechip_swapchannel_global(
        input->addr,
        output->addr,
        input->shape,
        sc_param.order,
        tpu_type_convert(input->dtype));
}

// local api function
