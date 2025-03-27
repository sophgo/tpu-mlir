#include "param_parser.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

const char *getInt2DType(int num) {
  switch (num) {
  case 1:
    return "float32";
  case 2:
    return "float16";
  case 3:
    return "int8";
  case 4:
    return "uint8";
  default:
    // When the input number is not supported, return NULL as an error
    // indication.
    return NULL;
  }
}

void inference_preprocess(void *param, int param_size,
                          const int (*input_shapes)[MAX_SHAPE_DIMS],
                          const int *input_dims, const float **inputs,
                          float **outputs) {
  PARSE_PARAM(preprocess, preprocess_param, param);
  float scale = preprocess_param.scale;
  float mean = preprocess_param.mean;
  int dtype = preprocess_param.type;
  int elem_num = 1;
  for (int i = 0; i < input_dims[0]; ++i) {
    elem_num *= input_shapes[0][i];
  }
  const char *type_str = getInt2DType(dtype);
  if (strcmp(type_str, "int8") == 0) {
    int8_t min_int8 = -128;
    int8_t max_int8 = 127;
    for (int i = 0; i < elem_num; ++i) {
      uint8_t input_int8 = (uint8_t)(inputs[0][i]);
      int32_t temp = (int32_t)(input_int8 - (uint8_t)(mean)) * (int32_t)(scale);
      if (temp < min_int8) {
        temp = min_int8;
      } else if (temp > max_int8) {
        temp = max_int8;
      }
      outputs[0][i] = (float)temp;
    }
  } else if (strcmp(type_str, "float32") == 0 ||
             strcmp(type_str, "float16") == 0) {
    for (int i = 0; i < elem_num; ++i) {
      outputs[0][i] = (inputs[0][i] - mean) * scale;
    }
  } else {
    fprintf(stderr, "Unknown data type: %s\n", type_str);
    return;
  }
}
