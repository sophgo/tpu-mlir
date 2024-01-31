#include <stdbool.h>
#include "param_parser.h"

void inference_crop(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
  const int* input_dims, const float** inputs, float** outputs) {
  PARSE_PARAM(crop, crop_param, param);
  int out_num = 1;
  for (int i = 0; i < input_dims[0] - 2; ++i) {
    out_num *= input_shapes[0][i];
  }
  const int hnew = crop_param.hnew;
  const int wnew = crop_param.wnew;
  const int hold = input_shapes[0][input_dims[0] - 2];
  const int wold = input_shapes[0][input_dims[0] - 1];
  for (int n = 0; n < out_num; ++n) {
    for (int i = 0; i < hnew; ++i) {
      int iold = i + crop_param.hoffset;
      for (int j = 0; j < wnew; ++j) {
        int jold = j + crop_param.woffset;
        outputs[0][n * hnew * wnew + i * wnew + j] = inputs[0][n * hold * wold + iold * wold + jold];
      }
    }
  }
}

void shape_inference_crop(void* param, int param_size, const int (*input_shapes)[MAX_SHAPE_DIMS],
  const int* input_dims, int (*output_shapes)[MAX_SHAPE_DIMS], int* output_dims) {
  PARSE_PARAM(crop, crop_param, param);
  output_dims[0] = input_dims[0];
  for (int i = 0; i < input_dims[0] - 2; ++i) {
    output_shapes[0][i] = input_shapes[0][i];
  }
  output_shapes[0][input_dims[0] - 2] = crop_param.hnew;
  output_shapes[0][input_dims[0] - 1] = crop_param.wnew;
}
