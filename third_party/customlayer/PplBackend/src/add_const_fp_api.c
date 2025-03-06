#include <string.h>
#include "tpu_utils.h"
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"

extern int add_tiling(global_addr_t ptr_dst, global_addr_t ptr_src, float rhs, int N, int C, int H,
               int W, bool relu, int dtype);

void api_addconst_global(const global_tensor_spec_t *input,
                         global_tensor_spec_t *output,
                         const void *param) {
  PARSE_PARAM(addconst, addconst_param, param);
  add_tiling(output->addr, input->addr, addconst_param.b_val, input->shape[0],
             input->shape[1], input->shape[2],
             input->shape[3], false, input->dtype);
}
