#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "backend_custom_param.h"

// parse param function
static crop_param_t crop_parse_param(const void* param) {
    crop_param_t crop_param = {0};
    crop_param.hoffset = ((custom_param_t*)param)[0].int_t;
    crop_param.woffset = ((custom_param_t*)param)[1].int_t;
    crop_param.hnew = ((custom_param_t*)param)[2].int_t;
    crop_param.wnew = ((custom_param_t*)param)[3].int_t;
    return crop_param;
}

// shape infer function
void shape_infer_crop(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    crop_param_t crop_param = crop_parse_param(param);
    output->dtype = input->dtype;
    output->dims = 2;
    output->shape[0] = crop_param.hnew;
    output->shape[1] = crop_param.wnew;
    output->elem_num = input->elem_num;
}

// global api function
void api_crop_global(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    crop_param_t crop_param = crop_parse_param(param);
    tpu_impl_crop_global(
        input->addr,
        output->addr,
        input->shape,
        crop_param.hoffset,
        crop_param.woffset,
        crop_param.hnew,
        crop_param.wnew,
        tpu_type_convert(input->dtype));
}

// local api function
