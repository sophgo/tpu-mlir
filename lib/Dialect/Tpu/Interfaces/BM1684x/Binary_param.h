#pragma once
#include "tpu_mlir/Backend/BM168x/BM1684x.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t input_A_global_addr;
  uint64_t input_B_global_addr;
  uint64_t output_global_addr;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  int scale_A;
  int scale_B;
  int rshift_A;
  int rshift_B;
  int if_relu;
  DATA_TYPE_T dtype_A;
  DATA_TYPE_T dtype_B;
  int round_mode;
} eltwise_fixed_global_param_t;

typedef struct {
  uint64_t *input_global_addr;
  uint64_t output_global_addr;
  uint64_t mask_global_addr;
  int input_num;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  int *coeff;
  int need_mask;
  int *mask_index;
  int if_relu;
  DATA_TYPE_T dtype;
} eltwise_float_global_param_t;

typedef struct {
  uint32_t *input_local_addr;
  uint32_t output_local_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  int *input_local_cstride;
  int *scale_weight;
  int *rshift;
  DATA_TYPE_T *input_dtype;
  int input_num;
  int if_relu;
  int round_mode;
} eltwise_fixed_local_param_t;

typedef struct {
  uint32_t *input_local_addr;
  uint32_t output_local_addr;
  uint32_t buffer_local_addr;
  int input_num;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  float *coeff;
  int *input_local_cstride;
  int if_relu;
  DATA_TYPE_T dtype;
} eltwise_float_local_param_t;

typedef struct bcbinary_common_spec {
    int32_t binary_type;
    int32_t if_relu;
    float relu_upper_limit;
    int32_t scale_A;
    int32_t scale_B;
    int32_t rshift_A;
    int32_t rshift_B;
} bcbinary_common_spec_t ;

typedef struct bcbinary_local_spec {
    bcbinary_common_spec_t common;
    uint32_t buffer_addr;
} bcbinary_local_spec_t;

typedef struct bcbinary_local_param {
    bcbinary_local_spec_t spec;
    int32_t A_is_coeff;
    int32_t B_is_coeff;
} bcbinary_local_param_t;

#ifdef __cplusplus
}
#endif
