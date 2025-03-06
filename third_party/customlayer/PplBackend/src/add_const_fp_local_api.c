#include <string.h>
#include "tpu_impl_custom_ops.h"
#include "param_parser.h"
#include "tpu_utils.h"

extern int addconst_local(local_addr_t ptr_dst, local_addr_t ptr_src, float rhs,
                          int N, int C, int H, int W, bool relu, int dtype);

void api_addconst_local(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *input,
    local_tensor_spec_t *output,
    const void *param) {
  int shape[4];
  parse_input_slice_shape(sec_info, input, shape);
  PARSE_PARAM(addconst, addconst_param, param);
  addconst_local(output->addr, input->addr, addconst_param.b_val, shape[0],
             shape[1], shape[2], shape[3], false, input->dtype);
}


