#include <string.h>
#include <assert.h>
#include "param_parser.h"

void inference_swapchannel(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
  const int* input_dims, const float** inputs, float** outputs) {
  PARSE_PARAM(swapchannel, sc_param, param);
  int in_num = 1;
  for (int i = 2; i < input_dims[0]; ++i) {
    in_num *= input_shapes[0][i];
  }
  int N = input_shapes[0][0];
  int C = input_shapes[0][1];
  assert(C == 3);
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int x = 0; x < in_num; ++x) {
        memcpy(outputs[0] + n * C * in_num + sc_param.order[c] * in_num,
               inputs[0] + n * C * in_num + c * in_num, in_num * sizeof(float));
      }
    }
  }
}
