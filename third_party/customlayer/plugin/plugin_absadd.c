#include <stdbool.h>
#include <math.h>
#include "param_parser.h"

void inference_absadd(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
  const int* input_dims, const float** inputs, float** outputs) {
  PARSE_PARAM(absadd, absadd_param, param);
  int elem_num = 1;
  for (int i = 0; i < input_dims[0]; ++i) {
    elem_num *= input_shapes[0][i];
  }
  for (int i = 0; i < elem_num; ++i) {
    outputs[0][i] = fabsf(inputs[0][i]) + absadd_param.b_val;
  }
}

bool local_gen_support_absadd(void* param, int param_size) {
  return true;
}
