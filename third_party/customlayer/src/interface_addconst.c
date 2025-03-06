#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"


// type infer function (default, no need to override)

// global api function
int api_addconst_local_bfsz(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *input_spec,
    local_tensor_spec_t *output_spec,
    const void* param)
{
    int shape[4];
    parse_input_slice_shape(sec_info, input_spec, shape);
    return get_addconst_local_bfsz(
            shape,
            tpu_type_convert(input_spec->dtype));
}

