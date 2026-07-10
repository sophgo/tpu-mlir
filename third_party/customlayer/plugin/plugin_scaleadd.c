#include <stdbool.h>
#include "param_parser.h"

// CPU reference inference, used by the mlir interpreter / verification.
// Multi-input  : inputs[0] = A, inputs[1] = B
// Multi-output : outputs[0] = A * scale, outputs[1] = A * scale + B
void inference_scaleadd(void* param, int param_size,
    const int (*input_shapes)[MAX_SHAPE_DIMS],
    const int* input_dims, const float** inputs, float** outputs) {
  PARSE_PARAM(scaleadd, scaleadd_param, param);
  int elem_num = 1;
  for (int i = 0; i < input_dims[0]; ++i) {
    elem_num *= input_shapes[0][i];
  }
  for (int i = 0; i < elem_num; ++i) {
    outputs[0][i] = inputs[0][i] * scaleadd_param.scale;
    outputs[1][i] = outputs[0][i] + inputs[1][i];
  }
}

// MANDATORY for multi-output: the fallback in Top/Custom.cpp only fills
// output[0]. Fill EVERY output's shape here, otherwise outputs[1..] are
// left unset and shape inference is wrong.
void shape_inference_scaleadd(void* param, int param_size,
    const int (*input_shapes)[MAX_SHAPE_DIMS],
    const int* input_dims,
    int (*output_shapes)[MAX_SHAPE_DIMS], int* output_dims) {
  PARSE_PARAM(scaleadd, scaleadd_param, param);
  // both outputs share input[0]'s shape
  for (int o = 0; o < 2; ++o) {
    output_dims[o] = input_dims[0];
    for (int i = 0; i < input_dims[0]; ++i) {
      output_shapes[o][i] = input_shapes[0][i];
    }
  }
}

bool local_gen_support_scaleadd(void* param, int param_size) {
  // this example only provides the global (group=false) PPL kernel
  return false;
}
