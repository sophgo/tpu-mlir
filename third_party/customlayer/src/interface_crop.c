#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"

// type infer function
void type_infer_crop(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    PARSE_PARAM(crop, crop_param, param);
    output->dtype = input->dtype;
    output->dims = input->dims;
    output->shape[0] = input->shape[0];
    output->shape[1] = input->shape[1];
    output->shape[2] = crop_param.hnew;
    output->shape[3] = crop_param.wnew;
    output->elem_num = input->elem_num;
}

// global api function
void api_crop_global(
    const global_tensor_spec_t *input,
    global_tensor_spec_t *output,
    const void *param) {
    PARSE_PARAM(crop, crop_param, param);
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
