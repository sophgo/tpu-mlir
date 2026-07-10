#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"

extern int scale_add_tiling(global_addr_t ptr_out0, global_addr_t ptr_out1,
                            global_addr_t ptr_in0, global_addr_t ptr_in1,
                            float scale, int N, int C, int H, int W, int dtype);

// `input` / `output` are ARRAYS passed by backend_api_<name>_global (see
// backend_helper.h IMPL_CUSTOM_PPL_API_GLB). Index them for multi-input /
// multi-output: input[0]=A, input[1]=B ; output[0]=A*scale, output[1]=A*scale+B.
void api_scaleadd_global(const global_tensor_spec_t *input,
                         global_tensor_spec_t *output,
                         const void *param) {
  PARSE_PARAM(scaleadd, scaleadd_param, param);
  scale_add_tiling(output[0].addr, output[1].addr, input[0].addr, input[1].addr,
                   scaleadd_param.scale, input[0].shape[0], input[0].shape[1],
                   input[0].shape[2], input[0].shape[3], input[0].dtype);
}
