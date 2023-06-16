#include "tpu_utils.h"
#include "api_swapchannel.h"
#include "nodechip_swapchannel.h"

// global api function
void api_swapchannel_global(
    global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    swapchannel_param_t *param)
{
    nodechip_swapchannel_global(
        input->addr,
        output->addr,
        input->shape,
        param->order,
        tpu_type_convert(input->dtype));
}
