#include <stdbool.h>
#include <math.h>
#include "param_parser.h"

void inference_ceiladd(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
  const int* input_dims, const float** inputs, float** outputs) {
  PARSE_PARAM(ceiladd, ceiladd_param, param);
  int elem_num = 1;
  for (int i = 0; i < input_dims[0]; ++i) {
    elem_num *= input_shapes[0][i];
  }
  for (int i = 0; i < elem_num; ++i) {
    outputs[0][i] = ceilf(inputs[0][i]) + ceiladd_param.b_val;
  }
}

bool local_gen_support_ceiladd(void* param, int param_size) {
  return true;
}
