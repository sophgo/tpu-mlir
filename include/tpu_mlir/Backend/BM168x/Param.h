//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

// #include "tpu_mlir/Backend/Arch.h"

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------------------------------------------------
// Chip Definition for PPL
// -------------------------------------------------------------------
#define PPL_BM1684X "bm1684x"
#define PPL_BM1688 "bm1688"
#define PPL_BM1690 "bm1690"

// -------------------------------------------------------------------
// Constant Definition
// -------------------------------------------------------------------

#define MAX_TPU_DIM 65535
#define MAX_SHAPE_DIMS 8
#define MAX_SPLIT_OUTPUT_NUM 8

#define SUBNET_MODE_TPU 0
#define SUBNET_MODE_CPU 1
#define SUBNET_MODE_MERGE 2
#define SUBNET_MODE_SWITCH 3

#define MEM_TYPE_TPU (1 << 0)
#define MEM_TYPE_CPU (1 << 1)
#define MEM_TYPE_ALL (MEM_TYPE_TPU | MEM_TYPE_CPU)

#define GDMA_VALUE_DIR_S2L 0
#define GDMA_VALUE_DIR_L2S 1
#define GDMA_VALUE_DIR_S2S 2
#define GDMA_VALUE_DIR_L2L 3

// -------------------------------------------------------------------
// Enum Definition
// -------------------------------------------------------------------

typedef enum {
  STORAGE_MODE_1N_FP32 = 0,
  STORAGE_MODE_1N_INT8 = 1,
  STORAGE_MODE_1N_INT16 = 2,
  STORAGE_MODE_2N_INT16 = 3,
  STORAGE_MODE_4N_INT8 = 4,
  STORAGE_MODE_2IC_FP32 = 5, // special for 2IC weight
  STORAGE_MODE_4N_4IC_4OC = 6,
  STORAGE_MODE_4N_INT16 = 7,
  STORAGE_MODE_UNINITILIZED,
  STORAGE_MODE_END
} TENSOR_STORAGE_MODE;

typedef enum {
  STORE_MODE_1N = 0,
  STORE_MODE_2N = 1,
  STORE_MODE_4N = 2,
  STORE_3IC = 3,
  // if need to support more store mode, pls add below
} STORE_MODE_T;

typedef enum {
  DTYPE_FP32 = 0,
  DTYPE_FP16 = 1,
  DTYPE_INT8 = 2,
  DTYPE_UINT8 = 3,
  DTYPE_INT16 = 4,
  DTYPE_UINT16 = 5,
  DTYPE_INT32 = 6,
  DTYPE_UINT32 = 7,
  DTYPE_BFP16 = 8,
  DTYPE_INT4 = 9,
  DTYPE_UINT4 = 10,
  DTYPE_FP20 = 11,
  DTYPE_F8E5M2 = 12,
  DTYPE_F8E4M3 = 13,
  DTYPE_UNKNOWN = -1,
} DATA_TYPE_T;

typedef enum {
  ROUND_INF = 0,  // 1.5 -> 2   -1.5 -> -2
  ROUND_UP = 1,   // 1.5 -> 2   -1.5 -> -1
  ROUND_DOWN = 2, // 1.5 -> 1   -1.5 -> -2
  ROUND_EVEN = 3, // 1.5 -> 2    2.5 -> 2
  ROUND_ODD = 4,  // 1.5 -> 1    0.5 -> 1
  ROUND_ZERO = 5, // 1.5 -> 1   -1.5 -> -1
  TRIM_ZERO = 6,  // 1.6 -> 1   -1.6 -> -1
  TRIM_INF = 7,   // 1.4 -> 2   -1.4 -> -2
  TRIM_UP = 8,    // 1.4 -> 2   -1.6 -> -1
  TRIM_DOWN = 9,  // 1.6 -> 1   -1.4 -> -2
} ROUND_MODE_T;

typedef enum {
  BMNET_NEURON = 0,       // Addr align, h*w align
  BMNET_COEFF = 1,        // Addr unalign, h*w compact
  BMNET_COEFF_NEURON = 2, // Addr align, h*w compact
  BMNET_COEFF_FC = 3,
  BMNET_COEFF_WINOGRAD = 4,
  BMNET_NEURON_FC = 5,
  BMNET_NEURON_CONST = 6, // Addr align, h*w align
  BMNET_NEURON_SHAPE = 7,
  BMNET_NEURON_CPU = 8,
  BMNET_NEURON_ARRAY = 9,
  BMNET_NEURON_FLOW = 10,
  BMNET_NEURON_3IC = 11,
  BMNET_CPU_CONST = 12,
  BMNET_COEFF_ALIGN = 13, // Addr align, h*w align
  TENSOR_TYPE_NUM,
  TENSOR_UNKNOWN = -1,
} TENSOR_TYPE_T;

typedef enum {
  ELTWISE_PRODUCT = 0,
  ELTWISE_ADD = 1,
  ELTWISE_MAX = 2,
} ELTWISE_OPCODE_T;

typedef enum {
  ACTIVE_TANH = 0,
  ACTIVE_SIGMOID = 1,
  ACTIVE_RELU = 2,
  ACTIVE_EXP = 3,
  ACTIVE_ELU = 4,
  ACTIVE_SQRT = 5,
  ACTIVE_SQUARE = 6,
  ACTIVE_RSQRT = 7,
  ACTIVE_ABSVAL = 8,
  ACTIVE_LN = 9,
  ACTIVE_ROUND = 10,
  ACTIVE_CEIL = 11,
  ACTIVE_FLOOR = 12,
  ACTIVE_SIN = 13,
  ACTIVE_COS = 14,
  ACTIVE_IS_FINITE = 15,
  ACTIVE_MISH = 16,
  ACTIVE_SWISH = 17,
  ACTIVE_HSWISH = 18,
  ACTIVE_SILU = 19,
  ACTIVE_ARCSIN = 20,
  ACTIVE_ARCCOS = 21,
  ACTIVE_ARCSINH = 22,
  ACTIVE_ARCCOSH = 23,
  ACTIVE_ARCTANH = 24,
  ACTIVE_SINH = 25,
  ACTIVE_COSH = 26,
  ACTIVE_TAN = 27,
  ACTIVE_SIGN = 28,
  ACTIVE_GELU = 29,
  ACTIVE_ERF = 30,
  ACTIVE_HSIGMOID = 31,
  ACTIVE_LOG_SIGMOID = 32,
  ACTIVE_SOFT_PLUS = 33,
  ACTIVE_SOFT_SIGN = 34,
} active_type_t;

typedef enum {
  BINARY_ADD = 0,
  BINARY_SUB = 1,
  BINARY_MUL = 2,
  BINARY_DIV = 3,
  BINARY_MAX = 4,
  BINARY_MIN = 10000,
  BINARY_GT = 10001,
  BINARY_GE = 10002,
  BINARY_LT = 10003,
  BINARY_LE = 10004,
  BINARY_EQ = 10005,
  BINARY_NE = 10006,
  BINARY_SQUARED_DIFF = 10007,
  BINARY_FLOOR_MOD = 10008,
  BINARY_FLOOR_DIV = 10009,
  BINARY_LOGIC_AND = 10010,
  BINARY_LOGIC_OR = 10011,
  BINARY_LOGIC_XOR = 10012,
  BINARY_BIT_AND = 10013,
  BINARY_BIT_OR = 10014,
  BINARY_BIT_XOR = 10015,
} binary_type_t;

typedef enum {
  CAFFE_SUPPORT = 0,
  TENSORFLOW_SUPPORT = 1,
  CAFFE_NEAREST = 2,
  TENSORFLOW_NEAREST = 3,
  PYTORCH_SUPPORT = 4,
  PYTORCH_NEAREST = 5,
  OPENCV_BILINEAR = 6,
  ONNX_NEAREST = 7,
} PLATFORM_SUPPORT;

typedef enum {
  SG_REDUCE_MEAN = 0,
  SG_REDUCE_SUM = 1,
  SG_REDUCE_MAX = 2,
  SG_REDUCE_MIN = 3,
  SG_REDUCE_PROD = 4,
  SG_REDUCE_L2 = 5,
  SG_REDUCE_L1 = 6,
} sg_reduce_method_t;

typedef enum {
  ARG_MAXT = 0,
  ARG_MINT = 1,
} arg_method_t;

typedef enum {
  UNARY_F32_ABS,
  UNARY_F32_ACOSH,
  UNARY_F32_ARCCOS,
  UNARY_F32_ARCSIN,
  UNARY_F32_ASINH,
  UNARY_F32_ATANH,
  UNARY_F32_CEIL,
  UNARY_F32_COS,
  UNARY_F32_COSH,
  UNARY_F32_COT,
  UNARY_F32_EXP,
  UNARY_F32_EXPM1,
  UNARY_F32_FLOOR,
  UNARY_F32_IPOWER,
  UNARY_F32_SQUARE,
  UNARY_F32_LOG,
  UNARY_F32_LOG1P,
  UNARY_F32_PRELU,
  UNARY_F32_PRELU_N,
  UNARY_F32_RELU,
  UNARY_F32_RELU_N,
  UNARY_F32_ELU,
  UNARY_F32_ROUND,
  UNARY_F32_RSQRT,
  UNARY_F32_SIGMOID,
  UNARY_F32_SIGN,
  UNARY_F32_SIN,
  UNARY_F32_SINH,
  UNARY_F32_SQRT,
  UNARY_F32_TAN,
  UNARY_F32_TANH,
  UNARY_F32_TO_I32,
  UNARY_F32_TRIM,
  UNARY_I32_TO_F32,
  UNARY_U32_TO_F32,
  UNARY_U8_TO_F32,
  UNARY_F32_MISH,
  UNARY_F32_SWISH,
  UNARY_F32_IS_FINITE,
  UNARY_F32_GELU,
} UNARY_FUNC_TYPE;
// -------------------------------------------------------------------
// Struct Definition
// -------------------------------------------------------------------

struct cmd_id_node;
typedef struct cmd_id_node CMD_ID_NODE;

typedef struct nnvlc_common_spec {
  int32_t do_compress;
  int32_t do_decompress;
  int32_t bias0;
  int32_t bias1;
  int32_t zero_guard;
} nnvlc_common_spec_t;

typedef struct bmcompiler_mem_info {
  uint64_t addr;
  uint64_t size;
  uint64_t offset;
} bm_mem_desc_t;
typedef struct bmcompiler_mem_info bm_device_mem_t;

typedef struct local_tensor_spec {
  uint64_t addr;
  int32_t dtype;
  int32_t dims;
  int32_t shape[MAX_SHAPE_DIMS];
  uint8_t consume_num;
  int *host_data;
  int elem_num;
} tensor_spec_t;

typedef struct stride {
  int64_t N, C, H, W;
} stride_4D_t;

typedef struct active_common_spec {
  int active_type;
  float coeffs[MAX_SHAPE_DIMS];
} active_common_spec_t;

typedef struct active_global_spec {
  active_common_spec_t common;
} active_global_spec_t;

typedef struct active_local_spec {
  active_common_spec_t common;
  uint32_t buffer_addr;
} active_local_spec_t;

typedef struct active_local_param {
  active_local_spec_t spec;
} active_local_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint32_t buffer_local_addr; // for local layer param
  int shape[MAX_SHAPE_DIMS];
  int shape_dim;
  int dtype;
  int active_type;
} active_param_t;

// use for constbinary
typedef struct constbinary_common_spec {
  float B_const_val;
  int B_dtype;
  int inversed;
  int binary_type;
  int if_relu;
  float relu_upper_limit;
  int scale_A;
  int rshift_A;
  float f8_scale_A;
  int zp_out;
} constbinary_common_spec_t;

typedef struct constbinary_global_spec {
  constbinary_common_spec_t common;
} constbinary_global_spec_t;

typedef struct constbinary_local_spec {
  constbinary_common_spec_t common;
  uint32_t buffer_addr;
} constbinary_local_spec_t;

typedef struct constbinary_local_param {
  constbinary_local_spec_t spec;
} constbinary_local_param_t;

typedef struct concat_common_spec {
  int input_num;
  int concat_axis;
} concat_common_spec_t;

typedef struct concat_global_spec {
  concat_common_spec_t common;
  int *is_st_concat_way;
} concat_global_spec_t;

typedef struct concat_local_spec {
  concat_common_spec_t common;
  int *is_st_concat_way;
} concat_local_spec_t;

typedef struct concat_local_param {
  concat_local_spec_t spec;
} concat_local_param_t;

typedef struct conv_common_spec {
  int32_t groups;
  int32_t input_c;
  int32_t output_c;
  int32_t kh;
  int32_t kw;
  int32_t stride_h;
  int32_t stride_w;
  int32_t dh;
  int32_t dw;
  int32_t pad_h_t;
  int32_t pad_h_b;
  int32_t pad_w_l;
  int32_t pad_w_r;
  int32_t has_bias;
  int32_t if_relu;
  float upper_limit;
  int32_t rshift;
  int32_t round_mode;
  int32_t is_asym;
  int32_t kzp_is_const;
  int32_t kzp_value;
  int32_t ipad_is_const;
  int32_t ipad_value;
  int32_t bias_sign; // For merged coeff
  int32_t use_3ic_optimize;
  int32_t weight_is_coeff;
  nnvlc_common_spec_t nnvlc_param;
} conv_common_spec_t;

typedef struct conv_global_spec {
  conv_common_spec_t common;
  /**
   * merge_coeff:
   *    0: Not merge and not reshape weight and bias
   *    1. reshape and merge weight and bias as (bias, weight) align to (4, 1)
   * bytes for depthwise_fix8b or (4, 64) bytes for conv_fix8b
   *    2. reshape and merge weight, bias and requant as has bias-(requant,
   * bias, weight) align to (64, 4, 1) bytes for depthwise_fix8b or (64, 4, 64)
   * bytes for conv_fix8b or no bias-(requant, weight) align to (64, 1) bytes
   * for depthwise_fix8b or (64, 64) bytes for conv_fix8b
   */
  int32_t merge_coeff;
  int32_t weight_is_tensor;
  int32_t using_multicore;
} conv_global_spec_t;

typedef struct conv_local_spec {
  conv_common_spec_t common;
  uint32_t buffer_local_addr;
  int32_t result_add;
  int32_t unused_ht_for_input;
  int32_t unused_hb_for_input;
  int32_t unused_wl_for_input;
  int32_t unused_wr_for_input;
  int32_t group_one_conv;
  int32_t with_requant;
  int32_t merge_coeff;

  // For dynamic inference
  uint32_t concat_c;
  int32_t concat_c_idx;
  int32_t reference_id;
} conv_local_spec_t;

typedef struct conv_local_param {
  conv_local_spec_t spec;
} conv_local_param_t;

typedef struct {
  uint64_t input_global_addr;
  uint64_t weight_global_addr;
  uint64_t bias_global_addr;
  uint64_t output_global_addr;
  int32_t input_shape[5]; // (n, ic, it, ih, iw)
  int32_t groups;
  int32_t output_c;
  int32_t kernel[3];
  int32_t stride[3];
  int32_t dilation[3];
  int32_t pad[6];
  int32_t has_bias;
  int32_t input_dtype;
  int32_t weight_dtype;
  int32_t bias_dtype;
  int32_t output_dtype;
  int32_t do_relu;
  float relu_limit;
  uint64_t kzp_global_addr;
  uint64_t pad_global_addr;
  bool kzp_is_const;
  bool pad_is_const;
  int32_t kzp_val;
  int32_t pad_val;
  int32_t kzp_dtype;
} conv3d_global_spec_t;

typedef struct conv3d_local_param {
  uint32_t input_local_addr;
  uint32_t weight_local_addr;
  uint32_t bias_local_addr;
  uint32_t buffer_local_addr;
  uint32_t output_local_addr;
  int32_t input_shape[5]; // (id, n, ic, ih, iw)
  int32_t groups;
  int32_t output_c;
  int32_t kernel[3];   // (kd, kh, kw)
  int32_t stride[3];   // (sd, sh, sw)
  int32_t dilation[3]; // (dd, dh, dw)
  int32_t pad[6];      // (df, db, ht, hb, wl, wr)
  int32_t has_bias;
  int32_t input_dtype;
  int32_t weight_dtype;
  int32_t bias_dtype;
  int32_t output_dtype;
  int32_t do_relu;
  float relu_limit;
  uint32_t kzp_local_addr;
  uint32_t pad_local_addr;
  bool kzp_is_const;
  bool pad_is_const;
  int32_t kzp_val;
  int32_t pad_val;
  int32_t kzp_dtype;
} conv3d_local_spec_t;

typedef struct {
  /* common param */
  uint64_t input_global_addr;
  uint64_t weight_global_addr;
  uint64_t bias_global_addr;
  uint64_t output_global_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];     // (kh, kw)
  int stride[2];     // (h, w)
  int dilation[2];   // (h, w)
  int pad[4];        // (h0, h1, w0, w1)
  int output_pad[2]; // (h, w)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  bool is_asym;
  unsigned char rshift;
  uint64_t kzp_global_addr;
  uint64_t pad_insert_global_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv_global_param_t;

typedef struct {
  /* common param */
  unsigned int input_local_addr;
  unsigned int weight_local_addr;
  unsigned int bias_local_addr;
  unsigned int buffer_local_addr;
  unsigned int output_local_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];   // (kh, kw)
  int stride[2];   // (h, w)
  int dilation[2]; // (h, w)
  int pad[4];      // (h0, h1, w0, w1)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  bool is_asym;
  unsigned char rshift;
  unsigned int kzp_local_addr;
  unsigned int pad_insert_local_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv_local_param_t;

typedef struct {
  /* common param */
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int input_shape[5]; // (n, ic, it, ih, iw)
  int groups;
  int output_c;
  int kernel[3];     // (kt, kh, kw)
  int stride[3];     // (t, h, w)
  int dilation[3];   // (t, h, w)
  int pad[6];        // (t0, t1, h0, h1, w0, w1)
  int output_pad[3]; // (t, h, w)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  unsigned long long kzp_global_addr;
  unsigned long long pad_insert_global_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv3d_global_param_t;

typedef struct {
  /* common param */
  unsigned int input_local_addr;
  unsigned int weight_local_addr;
  unsigned int bias_local_addr;
  unsigned int buffer_local_addr;
  unsigned int output_local_addr;
  int input_shape[5]; // (it, n, ic, ih, iw)
  int groups;
  int output_c;
  int kernel[3];   // (kt, kh, kw)
  int stride[3];   // (t, h, w)
  int dilation[3]; // (t, h, w)
  int pad[6];      // (t0, t1, h0, h1, w0, w1)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  unsigned int kzp_local_addr;
  unsigned int pad_insert_local_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv3d_local_param_t;

typedef struct bcbinary_common_spec {
  int32_t binary_type;
  int32_t if_relu;
  float relu_upper_limit;
  int32_t scale_A;
  int32_t scale_B;
  int32_t rshift_A;
  int32_t rshift_B;
  float f8_scale_A;
  float f8_scale_B;
  int32_t izp_A;
  int32_t izp_B;
  int32_t ozp;
} bcbinary_common_spec_t;

typedef struct bcbinary_local_spec {
  bcbinary_common_spec_t common;
  uint32_t buffer_addr;
} bcbinary_local_spec_t;

typedef struct bcbinary_local_param {
  bcbinary_local_spec_t spec;
  int32_t A_is_coeff;
  int32_t B_is_coeff;
} bcbinary_local_param_t;

typedef struct bcbinary_global_param {
  bcbinary_common_spec_t spec;
  int32_t A_is_coeff;
  int32_t B_is_coeff;
} bcbinary_global_param_t;

typedef struct binaryshift_spec {
  int32_t binary_op;
  int32_t rshift_num;
  int32_t b_is_const;
  int32_t b_const_val;
  int32_t inversed;
  int32_t round_mode;
  bool is_saturate;
} binaryshift_spec_t;

typedef struct binaryshift_local_spec {
  binaryshift_spec_t common;
  uint32_t buffer;
} binaryshift_local_spec_t;

typedef struct binaryshift_local_param {
  binaryshift_local_spec_t spec;
  int32_t a_is_coeff;
  int32_t b_is_coeff;
} binaryshift_local_param_t;

typedef struct binaryshift_global_param {
  binaryshift_spec_t spec;
  int32_t a_is_coeff;
  int32_t b_is_coeff;
} binaryshift_global_param_t;

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

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t requant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  int mul_value;
  int shift_value;
  int offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
  int reshaped_coeff;
  int zx_value;
  int round_mode;
} requant_int_param_t;

typedef struct gather_elements_global {
  int axis;
  int index_is_coeff; // use for dyn
  uint64_t intermediate_buffer_global_addr;
} gather_elements_global_param_t;

typedef struct gather_nd_global {
  int batch_dims;
  int const_val; // fill_value if index not found in input
} gather_nd_global_param_t;

typedef struct scatter_elements_global_spec {
  int data_dims;
  int indices_dims;
  int updates_dims;
  uint64_t intermediate_buffer_global_addr;
  int axis;
} scatter_elements_global_spec_t;

typedef struct index_select_common_spec {
  int axis;
  int index_is_coeff; // use for dyn
  int if_neg_index;
} index_select_common_spec_t;

typedef struct index_select_global_spec {
  index_select_common_spec_t common;
} index_select_global_spec_t;

typedef struct index_put_spec {
  int mode;
  int accumulate;
  uint64_t buffer_addr;
} index_put_spec_t;

typedef struct roi_align_spec {
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  int sampling_ratio;
  int position_sensitive;
  int align_corners;
  int plat_sp;
} roi_align_spec_t;

typedef struct {
  bool is_perchannel;
  int mul_value;
  int shift_value;
  int offset_value;
  int output_dtype;
  int mode;
  int reshaped_coeff;
  int zx_value;
  int round_mode;
} dyn_requant_int_common_param_t;

typedef struct {
  dyn_requant_int_common_param_t common;
  uint32_t buffer_local_addr;
} dyn_requant_int_local_param_t;

typedef struct {
  dyn_requant_int_common_param_t common;
} dyn_requant_int_global_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t dequant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  int scale_val;
  int shift_val;
  int offset_val;
  int mode;
  int lshift;
  DATA_TYPE_T input_dtype;
  DATA_TYPE_T output_dtype;
  int round_mode;
} dequant_int_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t slope_addr;
  uint64_t output_addr;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int channel_shared;
  float slope_val;
  int rshift_bit;
  float relu_limit;
  DATA_TYPE_T dtype;
} prelu_param_t;

typedef struct {
  int sel0_is_const;
  int sel1_is_const;
  float sel0_const_val;
  float sel1_const_val;
  uint64_t buffer_addr;
} select_common_spec_t;

typedef struct {
  float scale_val;
  int begin_axis;
  int end_axis;
  int log;
  int zero_point;
} softmax_common_param_t;

typedef struct {
  softmax_common_param_t common;
} softmax_global_param_t;

typedef struct {
  softmax_common_param_t common;
  uint32_t buffer_addr;
} softmax_local_param_t;

typedef struct {
  softmax_common_param_t common;
} softmax_tflite_fix8b_param_t;

typedef struct layer_norm_common_spec {
  int axis;
  float eps;
  int affine;
  int need_mean;
  int need_rstd;
} layer_norm_common_spec_t;

typedef struct layer_norm_global_spec {
  layer_norm_common_spec_t common;
} layer_norm_global_spec_t;

typedef struct layer_norm_local_spec {
  layer_norm_common_spec_t common;
  uint32_t buffer_addr;
} layer_norm_local_spec_t;

typedef struct rms_norm_common_spec {
  float eps;
  int affine;
} rms_norm_common_spec_t;

typedef struct rms_norm_global_spec {
  rms_norm_common_spec_t common;
} rms_norm_global_spec_t;

typedef struct rms_norm_local_spec {
  rms_norm_common_spec_t common;
  uint32_t buffer_addr;
} rms_norm_local_spec_t;

typedef struct instance_norm_common_spec {
  float eps;
  int affine;
} instance_norm_common_spec_t;

typedef struct instance_norm_global_spec {
  instance_norm_common_spec_t common;
} instance_norm_global_spec_t;

typedef struct instance_norm_local_spec {
  instance_norm_common_spec_t common;
  uint32_t buffer_addr;
} instance_norm_local_spec_t;

typedef struct group_norm_common_spec {
  int group_num;
  float eps;
  int affine;
} group_norm_common_spec_t;

typedef struct group_norm_local_param {
  group_norm_common_spec_t common;
  uint32_t buffer_addr;
} group_norm_local_param_t;

typedef struct group_norm_global_param {
  group_norm_common_spec_t common;
  int axis;
} group_norm_global_param_t;

typedef enum {
  GridSampleBilinear = 0,
  GridSampleNearest = 1,
} GridSampleInterpMode;

typedef enum {
  GridSampleZeros = 0,
  GridSampleBorder = 1,
  GridSampleReflection = 2,
} GridSamplePaddingMode;

typedef struct {
  uint64_t input_addr;
  uint64_t grid_addr;
  uint64_t output_addr;
  uint64_t buffer_addr;
  int input_n;
  int input_c;
  int input_d;
  int input_h;
  int input_w;
  int output_d;
  int output_h;
  int output_w;
  int dims;
  int align_corners;
  float mean;
  float scale;
  bool need_permute;
  GridSampleInterpMode interp_mode;
  GridSamplePaddingMode padding_mode;
  int dtype;
} grid_sample_global_param_t;

typedef struct tranpose_spec {
  uint64_t buffer_global_addr;
  uint32_t order[MAX_SHAPE_DIMS];
  uint32_t is_dynamic;
} transpose_spec_t;

typedef struct transpose_param {
  transpose_spec_t spec;
  int32_t if_getting_buffer_size;
  uint64_t *buffer_size_ptr;
  int num_core;
} transpose_param_t;

typedef struct shape_transpose_param {
  uint32_t order[MAX_SHAPE_DIMS];
} shape_transpose_param_t;
typedef struct reshape_spec {
  int32_t dims;
  int32_t shape[MAX_SHAPE_DIMS];
  int eu_align;
} reshape_spec_t;

typedef struct shape_reshape_param {
  int32_t dims;
  int32_t shape[MAX_SHAPE_DIMS];
} shape_reshape_param_t;
typedef struct {
  int pad[4][2];
  int type;
  float constant;
} pad_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t requant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
  int round_mode;
  int src_round_mode;
} requant_fp_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t dequant_addr;
  uint64_t buffer_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  bool has_buffer;
  float scale_value;
  int offset_value;
  int input_dtype;
  int output_dtype;
  int round_mode;
} dequant_fp_param_t;

typedef struct interp_common_spec {
  int pad_bag;
  int pad_end;
  bool align_corners;
  bool half_pixel_centers;
  int platform_sp;
} interp_common_spec_t;

typedef struct interp_global_spec {
  interp_common_spec_t common;
  int shape_is_fixed;
  int shape[MAX_SHAPE_DIMS];
  int dims;
  uint64_t buffer_addr;
} interp_global_spec_t;

typedef struct interp_global_param {
  interp_global_spec_t spec;
  int if_getting_buffer_size;
  uint64_t *buffer_size_ptr;
} interp_global_param_t;

typedef struct interp_local_spec {
  interp_common_spec_t common;
} interp_local_spec_t;

typedef struct cast_common_spec {
  int src_dtype;
  int dst_dtype;
  int round_mode;
} cast_common_spec_t;

typedef struct cast_global_spec {
  cast_common_spec_t common;
} cast_global_spec_t;

typedef struct cast_local_spec {
  cast_common_spec_t common;
  uint32_t buffer_addr;
} cast_local_spec_t;

typedef struct cast_local_param {
  cast_local_spec_t spec;
} cast_local_param_t;

typedef struct {
  bool is_perchannel;
  float scale_value;
  int offset_value;
  int output_dtype; // for fp16/bfp16 output, default(0) fp32 output
  int round_mode;
} dyn_dequant_fp_param_t;

typedef struct {
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int output_dtype;
  int mode;
  int round_mode;
  int src_round_mode;
} dyn_requant_fp_common_param_t;

typedef struct {
  dyn_requant_fp_common_param_t common;
  uint32_t buffer_local_addr;
} dyn_requant_fp_local_param_t;

typedef struct {
  dyn_requant_fp_common_param_t common;
} dyn_requant_fp_global_param_t;

typedef struct reduce_full_common_spec {
  int axis[MAX_SHAPE_DIMS];
  int axis_num;
  int method;
  float input_scale;
  float output_scale;
  int keep_dims; // used for dynamic compile
} reduce_full_common_spec_t;

typedef struct reduce_full_global_spec {
  reduce_full_common_spec_t common;
  uint64_t buffer_addr;
} reduce_full_global_spec_t;

typedef struct reduce_full_global_param {
  reduce_full_global_spec_t spec;
  int if_getting_buffer_size;
  uint64_t *buffer_size_ptr;
} reduce_full_global_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t slope_addr;
  uint64_t output_addr;
  int32_t input_n;
  int32_t input_c;
  int32_t input_h;
  int32_t input_w;
  int32_t channel_shared;
  float slope_val;
  int32_t rshift_bit;
  float relu_limit;
  DATA_TYPE_T dtype;
} leakyrelu_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int size;
  float alpha;
  float beta;
  float k;
  int dtype;
} lrn_global_param_t;

typedef struct {
  float upper_limit;
  float slope_val;
  int is_channel_shared;
  int rshift_bit;
  int round_mode;
} prelu_spec_t;

typedef struct {
  uint64_t input_addr;
  uint64_t table_addr;
  uint64_t output_addr;
  unsigned int buffer_addr; // used only for local layer
  int shape[MAX_SHAPE_DIMS];
  int shape_dim;
  int table_length;
  int input_dtype;
  int table_dtype;
  int output_dtype;
  int is_local_layer;
} lut_param_t;

typedef struct {
  int output_dtype;
  int is_local_layer;
} lut_common_param_t;

typedef struct {
  lut_common_param_t common;
  unsigned int buffer_addr; // used only for local layer
} dyn_lut_local_param_t;

typedef struct {
  lut_common_param_t common;
} dyn_lut_global_param_t;

typedef struct {
  unsigned int buffer_addr; // used only for local layer
  int output_dtype;
  int is_local_layer;
} dyn_lut_param_t;

typedef struct fc_global_spec {
  /* common param of float and fixed */
  int32_t R_transpose;
  int32_t have_bias;
  int32_t if_relu;
  float relu_limit;
  /* quantize param */
  int32_t rshift;
  int32_t is_asymmetric;
  int32_t rzp_const_val;
  int32_t rzp_is_const;
  int32_t izp_const_val;
  /* requantize param */
  int32_t requant_mode; // mode < 0 means no requantize
  int32_t mul_val;
  int32_t shift_val;
  int32_t offset_val;
  int32_t round_mode;
  int32_t fuse_rq;
  int if_getting_buffer_size;
  uint64_t *buffer_size_ptr;
  int32_t need_buffer;
} fc_global_spec_t;

typedef struct a16_matmul_spec {
  bool has_bias;
  bool sign;
  bool R_trans;
  int weight_bits;
  bool has_zp;
  int q_group_size;
} a16_matmul_spec_t;

typedef struct batch_matmul_common_spec {
  int Y_dtype;
  int L_trans;
  int R_trans;
  bool R_zp_is_const;
  int R_zp_const_val;
  int izp_const_val;
  bool has_bias;
  bool hdim_is_batch;
  bool do_relu;
  float upper_limit;
  /* requant param */
  int requant_mode; // mode < 0 means no requantize
  int mul_val;
  int shift_val;
  int offset_val;
  int round_mode;
  int left_reuse;
  bool fuse_rq;
} batch_matmul_common_spec_t;

typedef struct batch_matmul_global_spec {
  batch_matmul_common_spec_t common;
} batch_matmul_global_spec_t;

typedef struct batch_matmul_local_spec {
  batch_matmul_common_spec_t common;
  unsigned int buffer_addr;
} batch_matmul_local_spec_t;

typedef struct batch_matmul_local_param {
  batch_matmul_local_spec_t spec;
} batch_matmul_local_param_t;

typedef struct {
  int block_sizes[2];
  int in_is_nchw;
  int out_is_nchw;
  int is_inversed;
  int is_crd_mode;
  int swap_cr;
} depth2space_common_spec_t;

typedef struct {
  int size;
  int if_relu;
} upsample_spec_t;

typedef struct {
  uint64_t bottom_global_offset;
  uint64_t bottom_mask_global_offset;
  uint64_t top_global_offset;
  int bottom_global_N;
  int bottom_c;
  int bottom_h;
  int bottom_w;
  int top_c;
  int top_h;
  int top_w;
} upsamplemask_param_t;

typedef struct {
  int dims;
  int axis;
} reverse_global_param_t;

typedef struct shape_reverse_param {
  int dims;
  int axis;
} shape_reverse_param_t;
typedef struct {
  depth2space_common_spec_t common;
} depth2space_global_spec_t;

typedef struct pixel_norm_common_spec {
  float eps;
  int affine;
  float scale;
} pixel_norm_common_spec_t;

typedef struct pixel_norm_global_spec {
  pixel_norm_common_spec_t common;
} pixel_norm_global_spec_t;

typedef struct pixel_norm_local_spec {
  pixel_norm_common_spec_t common;
  uint32_t buffer_addr;
} pixel_norm_local_spec_t;

typedef struct {
  int axis;
  int axis_num;
  int has_bias;
  int if_relu;
  float relu_upper_limit;
  int scale_sign;
  int bias_sign;
  int merge_weight_bias;
  int round_mode;
  int version;
} scale_global_spec_t;

typedef struct {
  int shape_dim;
  int table_length;
} scalelut_param_t;

typedef struct {
  int data_dims;
  int indices_dims;
  int updates_dims;
  uint64_t intermediate_buffer_global_addr;
  bool with_hw_trans;
} scatter_nd_global_param_t;

typedef struct {
  int group;
} shuffle_channel_param_t;

typedef struct strideslice_common_spec {
  int begin_mask;
  int end_mask;
  int begin_index[MAX_SHAPE_DIMS];
  int end_index[MAX_SHAPE_DIMS];
  int strides[MAX_SHAPE_DIMS];
} strideslice_common_spec_t;

typedef struct strideslice_global_spec {
  strideslice_common_spec_t common;
  int shape_size;
  int ellipsis_mask;
  int new_axis_mask;
  int shrink_axis_mask;
  bool is_dynamic;
  bool begin_as_tensor;
  bool end_as_tensor;
  bool stride_as_tensor;
} strideslice_global_spec_t;

typedef struct strideslice_local_spec {
  strideslice_common_spec_t common;
  int buffer_addr;
} strideslice_local_spec_t;

typedef struct squeeze_dims_common_spec {
  int axis_list[MAX_SHAPE_DIMS];
  int axis_num;
} squeeze_dims_common_spec_t;

typedef struct squeeze_dims_global_spec {
  squeeze_dims_common_spec_t common;
} squeeze_dims_global_spec_t;
typedef struct shape_slice_param {
  int begin_index[MAX_SHAPE_DIMS];
  int end_index[MAX_SHAPE_DIMS];
  int stride[MAX_SHAPE_DIMS];
  int shape_size;
  int begin_mask;
  int end_mask;
  int shrink_axis_mask;
  int new_axis_mask;
  int ellipsis_mask;
  int is_dynamic;
} shape_slice_param_t;

typedef struct {
  /*common param*/
  int if_relu;
  float relu_upper_limit;
  int is_scale_coeff;
  int is_bias_coeff;
  int input_num;
  int merge_weight_bias;

  /*param for float*/
  int scale_shape[4];

  /*param for fixed*/
  unsigned int buffer_local_addr;
  int is_shift_coeff;
  int round_mode;
  int version;
  int bias_dtype;
} scale_local_spec_t;

typedef struct split_spec {
  int axis;
  int split_size[MAX_SPLIT_OUTPUT_NUM];
  int split_num;
  uint64_t buffer_addr;
  int input_num; // =2 means split_size is dynamic
} split_spec_t;

typedef struct swap_dim_spec {
  int axis_num;
  int axis_list[MAX_SHAPE_DIMS];
  int offset_list[MAX_SHAPE_DIMS];
  // int offset[MAX_SHAPE_DIMS];
} swap_dim_spec_t;

typedef struct swap_channel_param {
  int order[3];
  int shape_dim;
} swap_channel_param_t;

typedef struct {
  int tile_coeff[MAX_SHAPE_DIMS];
  int type;
} tile_common_spec_t;

typedef struct {
  tile_common_spec_t common;
  uint64_t buffer_addr;
} tile_global_spec_t;

typedef struct {
  tile_global_spec_t spec;
  int coeff_is_fixed;
  int input_is_coeff;
  int input_shape[MAX_SHAPE_DIMS];
  int input_dims;
  int dtype;
} tile_global_param_t;

typedef struct {
  int tile_axis;
  int tile_num;
  int type;
} tile_1d_global_param_t;

typedef struct {
  tile_common_spec_t common;
} tile_local_spec_t;

typedef struct {
  int k;
  int dim;
  int descending;
  uint64_t buffer_val_addr;
  uint64_t buffer_idx_addr;
} topk_spec_t;

typedef struct attention_common_spec {
  int head;
  float scale;
  int hasbias;
  int hasmusk;
  int input_num;
  int dim;
  int quant_param[16];
  int weight_reshape;
} attention_common_spec_t;

typedef struct attention_global_spec {
  attention_common_spec_t common;
} attention_global_spec_t;

typedef struct attention_local_spec {
  attention_common_spec_t common;
  uint32_t buffer_addr;
} attention_local_spec_t;

typedef struct flash_attention_common_spec {
  int batch;
  int q_head;
  int kv_head;
  int dim;
  int mq;
  int mk;
  float scale;
  int hasmask;
} flash_attention_common_spec_t;

typedef struct flash_attention_global_spec {
  flash_attention_common_spec_t common;
} flash_attention_global_spec_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  unsigned int buffer_addr; // only used for local layer
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int scale_val;
  int rshift_num;
  DATA_TYPE_T input_dtype;
  DATA_TYPE_T scale_dtype;
  DATA_TYPE_T output_dtype;
  ROUND_MODE_T round_mode;
} mulshift_param_t;

typedef struct {
  int scale_val;
  int rshift_num;
  DATA_TYPE_T scale_dtype;
  DATA_TYPE_T output_dtype;
  ROUND_MODE_T round_mode;
} dyn_mulshift_common_param_t;

typedef struct {
  dyn_mulshift_common_param_t common;
  unsigned int buffer_addr;
} dyn_mulshift_local_param_t;

typedef struct {
  dyn_mulshift_common_param_t common;
} dyn_mulshift_global_param_t;

typedef struct pooling_common_spec {
  int32_t kh;
  int32_t kw;
  int32_t pad_h_t;
  int32_t pad_h_b;
  int32_t pad_w_l;
  int32_t pad_w_r;
  int32_t stride_h;
  int32_t stride_w;
  int32_t dh;
  int32_t dw;
  int32_t is_global_pooling;
  int32_t is_avg_pooling;
  int32_t is_adaptive_pooling;
  int32_t avg_pooling_mode;
  /* for float */
  int32_t if_relu;
  float relu_limit;
  /* for fix8b */
  int32_t ceil_mode;
  int32_t round_mode;
  int32_t avg_pooling_quant_mode;
  int32_t max_pooling_with_mask; // 1: with mask 0: no mask
  int32_t multiplier;
  int32_t rshiftbits;
  /* asymmetric quantize */
  int32_t merge_requant;
  float rq_scale;
  float rq_offset;
  int32_t src_round_mode;
} pooling_common_spec_t;

typedef struct {
  int32_t buffer_addr;
  pooling_common_spec_t common;
} pooling_local_spec_t;

typedef struct pooling3d_spec {
  int64_t input_addr;
  int64_t output_addr;
  int32_t buffer_addr;
  int32_t input_shape[5];
  int32_t output_shape[5];
  int32_t *kernel;
  int32_t *stride;
  int32_t *dilation;
  int32_t *pad;
  bool is_avg_pooling;
  int32_t avg_pooling_mode;
  int32_t avg_rd_mode;
  /* for float */
  int32_t if_relu;
  float relu_limit;
  int32_t in_dtype;
  int32_t out_dtype;
  /* for fix8b */
  int32_t avg_pooling_quant_mode;
  bool merge_requant;
  float rq_scale;
  float rq_offset;
  int avg_src_rd_mode;
} pooling3d_spec_t;

typedef struct arg_common_spec {
  int axis;
  int method;
  int is_index_int32;
  int select_last_index;
  int need_val;
} arg_common_spec_t;

typedef struct arg_global_spec {
  arg_common_spec_t common;
} arg_global_spec_t;

typedef struct clip_spec {
  float min;
  float max;
  int if_relu;
} clip_spec_t;

typedef struct shape_clip_param {
  float min;
  float max;
} shape_clip_param_t;

typedef struct shape_pow_param {
  float exponent;
} shape_pow_param_t;

typedef struct rope_spec {
  uint64_t buffer_addr;
  int32_t mul1_saturation;
  int32_t mul2_saturation;
  int32_t add_saturation;
  int32_t mul1_shift;
  int32_t mul2_shift;
  int32_t add_shift;
  int32_t mul1_round_mode;
  int32_t mul2_round_mode;
  int32_t add_round_mode;
  int32_t is_permute_optimize;
} rope_param_t;

typedef struct where_spec {
  int order;
  uint64_t buffer_addr;
} where_spec_t;

typedef struct {
  int is_upper;
  int diagonal;
} triangularize_common_spec_t;

typedef struct conv3d_common_spec {
  int32_t groups;
  int32_t output_c;
  int32_t kernel[3];
  int32_t stride[3];
  int32_t dilation[3];
  int32_t pad[6];
  int32_t has_bias;
  int32_t input_dtype;
  int32_t weight_dtype;
  int32_t bias_dtype;
  int32_t output_dtype;
  int32_t do_relu;
  float relu_limit;
  bool kzp_is_const;
  bool pad_is_const;
  int32_t kzp_val;
  int32_t pad_val;
  int32_t kzp_dtype;
} conv3d_common_spec_t;

typedef enum {
  FLOAT32,
  UINT8,
} image_data_format_ext;

typedef enum {
  _601_limited,
  _601_full,
} formula_mode;

typedef struct {
  unsigned int batch;
  unsigned int width;
  unsigned int height;
  unsigned int src_format;
  unsigned int dst_format;
  image_data_format_ext output_data_format;
  formula_mode formula_mode;
  ROUND_MODE_T round_mode;
} yuv2rgb_formula_spec_t;

typedef struct dyn_conv3d_local_spec {
  conv3d_common_spec_t common;
  uint32_t kzp_local_addr;
  uint32_t pad_local_addr;
  uint32_t buffer_local_addr;
  // For dynamic inference
  uint32_t concat_c;
  int32_t concat_c_idx;
  int32_t reference_id;
} dyn_conv3d_local_spec_t;

typedef struct dyn_conv3d_local_param {
  dyn_conv3d_local_spec_t spec;
} dyn_conv3d_local_param_t;

typedef struct dyn_conv3d_global_spec {
  conv3d_common_spec_t common;
  uint64_t kzp_global_addr;
  uint64_t pad_global_addr;
  // For dynamic inference
  uint32_t concat_c;
  int32_t concat_c_idx;
  int32_t reference_id;
} dyn_conv3d_global_spec_t;

typedef struct dyn_conv3d_global_param {
  dyn_conv3d_global_spec_t spec;
} dyn_conv3d_global_param_t;

typedef struct dyn_deconv_common_spec {
  int groups;
  int output_c;
  int kernel[2];     // (kh, kw)
  int stride[2];     // (h, w)
  int dilation[2];   // (h, w)
  int pad[4];        // (h0, h1, w0, w1)
  int output_pad[2]; // (h, w)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  bool is_asym;
  unsigned char rshift;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} dyn_deconv_common_spec_t;

typedef struct dyn_deconv_local_spec {
  dyn_deconv_common_spec_t common;
  unsigned int buffer_local_addr;
  unsigned int kzp_local_addr;
  unsigned int pad_insert_local_addr;
} dyn_deconv_local_spec_t;

typedef struct dyn_deconv_global_spec {
  dyn_deconv_common_spec_t common;
  uint64_t kzp_global_addr;
  uint64_t pad_insert_global_addr;
  int output_pad[2]; // (h, w)
} dyn_deconv_global_spec_t;

typedef struct {
  bool is_perchannel;
  int scale_val;
  int shift_val;
  int offset_val;
  int mode;
  int lshift;
  int input_dtype;
  int output_dtype;
  int round_mode;
} dyn_dequant_int_common_spec_t;

typedef struct {
  dyn_dequant_int_common_spec_t common;
  uint32_t buffer_local_addr;
  uint32_t dequant_addr;
} dyn_dequant_int_local_spec_t;

typedef struct {
  dyn_dequant_int_common_spec_t common;
} dyn_dequant_int_global_spec_t;

typedef struct {
  bool bias;
  bool outputY;
  bool outputYh;
  int sequence;
  int batch;
  int xSize;
  int hSize;
  int batchMode;
  bool bidirectional;
  int numLayers;
  int dtype;
} gru_common_param_t;

typedef struct {
  gru_common_param_t common;
  uint64_t xGlobalAddr;
  uint64_t h0GlobalAddr;
  uint64_t yGlobalAddr;
  uint64_t hnGlobalAddr;
  uint64_t wGlobalAddr;
  uint64_t bGlobalAddr;
  uint64_t zGlobalAddr;
} dyn_glu_global_spec_t;

typedef struct {
  int size;
  float alpha;
  float beta;
  float k;
  int dtype;
} lrn_common_param_t;

typedef struct {
  lrn_common_param_t common;
} dyn_lrn_global_param_t;

typedef struct {
  int top_c;
  int top_h;
  int top_w;
} upsamplemask_common_param_t;

typedef struct {
  upsamplemask_common_param_t common;
} dyn_upsamplemask_global_spec_t;

typedef struct {
  int32_t output_shape[5];
  int32_t kernel[3];
  int32_t stride[3];
  int32_t dilation[3];
  int32_t pad[6];
  bool is_avg_pooling;
  int32_t avg_pooling_mode;
  int32_t avg_rd_mode;
  /* for float */
  int32_t if_relu;
  float relu_limit;
  int32_t in_dtype;
  int32_t out_dtype;
  /* for fix8b */
  int32_t avg_pooling_quant_mode;
  bool merge_requant;
  float rq_scale;
  float rq_offset;
  int32_t avg_src_rd_mode;
} pooling3d_common_param_t;

typedef struct {
  pooling3d_common_param_t common;
  int32_t buffer_addr;
} dyn_pooling3d_local_spec_t;

typedef struct {
  pooling3d_common_param_t common;
  int64_t buffer_addr;
} dyn_pooling3d_global_spec_t;

typedef struct {
  unsigned int filled_value;
  int dtype;
} constantfill_common_spec_t;

typedef struct shape_pack_param {
  int num_inputs;
  int axis;
} shape_pack_param_t;

typedef struct shape_cast_param {
  int output_dtype;
} shape_cast_param_t;

typedef struct {
  int axis_list[MAX_SHAPE_DIMS];
  int axis_num;
  int keep_dims;
  int reduce_method;
  float scale;
} shape_reduce_param_t;

typedef struct {
  int modulated;
  int deform_groups;
  int kh;
  int kw;
  int pad_h;
  int pad_w;
  int pad_h_after;
  int pad_w_after;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int mode;
  int offset_interleave;
  uint64_t buffer_addr;
} deform_gather_global_spec_t;

typedef struct nms_common_spec {
  float iou_threshold;
  float score_threshold;
  int keep_topk_per_class;
  int center_point_box;
  int input_num;
  int onnx_nms; // 1: onnx_nms
} nms_common_spec_t;

typedef struct dyn_nms_global_spec {
  nms_common_spec_t common;
  unsigned long long buffer_addr;
  int detected_box_num;
} dyn_nms_global_spec_t;

typedef struct pack_raw_spec {
  float white_level;
  float black_level;
  float threshold;
  int channel_order[4];
  int start_point[2];
} pack_raw_spec_t;

typedef struct depack_raw_spec {
  int pad[2];
  float white_level;
  float black_level;
  int channel_order[4];
  int start_point[2];
} depack_raw_spec_t;
#ifdef __cplusplus
}

typedef struct softmax_backward_cast_param {
  int axis;
} softmax_backward_param_t;

typedef struct weight_reorder_param {
  int reorder_mode;
} weight_reorder_param_t;

typedef struct batchnorm_train_param {
  float momentum;
  float eps;
} batchnorm_train_param_t;

typedef struct batchnorm_backward_param {
  int reserve;
} batchnorm_backward_param_t;

typedef struct layernorm_train_param {
  int axis;
  float eps;
} layernorm_train_param_t;

typedef struct layernorm_backward_param {
  int axis;
} layernorm_backward_param_t;

typedef struct embedding_backward_param {
  int window_size;
  bool is_index_int64;
} embedding_backward_param_t;

typedef struct {
  int32_t groups;
  int32_t ic;
  int32_t n;
  int32_t ih;
  int32_t iw;
  int32_t oc;
  int32_t oh;
  int32_t ow;
  int32_t kh;
  int32_t kw;
  int32_t sh;
  int32_t sw;
  int32_t dh;
  int32_t dw;
  int32_t pt;
  int32_t pb;
  int32_t pl;
  int32_t pr;
  bool has_bias;
} ConvBwdWeight_common_spec_t;

typedef struct {
  ConvBwdWeight_common_spec_t common;
} ConvBwdWeight_global_spec_t;

typedef struct {
  uint64_t buffer_addr;
  int axis;
  int descending;
  int is_argsort;
} sort_per_dim_param_t;

typedef struct {
  float eps;
  float momentum;
} mean_rstd_param_t;

typedef struct bdc_cpy_spec {
  int64_t reserved;
} bdc_cpy_spec_t;

typedef struct {
  ConvBwdWeight_common_spec_t common;
} ConvBwdWeight_local_spec_t;

typedef struct group_norm_train_global_param {
  group_norm_common_spec_t common;
  int axis;
} group_norm_train_global_param_t;
typedef struct {
  int length;
} LogicalAnd_param_t;

#define MAX_NUM_CHN (8)
typedef struct {
  int32_t num_of_chn;
  double std[MAX_NUM_CHN];
  double scale[MAX_NUM_CHN];
  double mean[MAX_NUM_CHN];
  int32_t multi[MAX_NUM_CHN];
  int32_t rshift[MAX_NUM_CHN];
  int32_t offset[MAX_NUM_CHN];
  int32_t in_zp;
  int32_t out_zp;
  int32_t round_mode;
} mean_std_scale_param_t;

typedef struct {
    uint32_t buffer_addr;
    uint32_t f32_param_addr;
    int32_t in_zp;
    int32_t out_zp;
    int32_t round_mode;
} mean_std_scale_local_param_t;

typedef struct {
  int32_t groups;
  int32_t ic;
  int32_t n;
  int32_t ih;
  int32_t iw;
  int32_t oc;
  int32_t oh;
  int32_t ow;
  int32_t kh;
  int32_t kw;
  int32_t sh;
  int32_t sw;
  int32_t dh;
  int32_t dw;
  int32_t pt;
  int32_t pb;
  int32_t pl;
  int32_t pr;
  int32_t insh;
  int32_t insw;
} Convbwd_common_spec_t;

typedef struct {
  Convbwd_common_spec_t common;
  uint64_t buffer_addr;
  int32_t grad_input_enable;
  int32_t grad_weight_enable;
  int32_t grad_bias_enable;
  int32_t use_multi_core;
} Convbwd_param_t;

typedef struct MaskRCNN_RPN_get_bboxes_common_spec {
  float delta2bbox_mean_0;
  float delta2bbox_mean_1;
  float delta2bbox_mean_2;
  float delta2bbox_mean_3;
  float delta2bbox_std_0;
  float delta2bbox_std_1;
  float delta2bbox_std_2;
  float delta2bbox_std_3;
  float delta2bbox_max_scalar_c;
  float iou_threshold;
  float conf_threshold;
} MaskRCNN_RPN_get_bboxes_common_spec_t;

typedef struct MaskRCNN_RPN_get_bboxes_global_param {
  MaskRCNN_RPN_get_bboxes_common_spec_t spec;
  unsigned long long global_buffer_0_batch_mlvl_scores;
  unsigned long long global_buffer_1_batch_mlvl_anchors;
  unsigned long long global_buffer_2_batch_mlvl_rpn_bbox_pred;
  unsigned long long global_buffer_3_batch_mlvl_proposals;
  unsigned long long global_buffer_4_batch_mlvl_ids;
  unsigned long long global_buffer_5_glb_buffer_tmp_scores_stretched;
  unsigned long long global_buffer_6_glb_buffer_ranked_scores;
  unsigned long long global_buffer_7_glb_buffer_rank_inds_int32;
  unsigned long long global_buffer_8_glb_buffer_rank_inds_u32;
  unsigned long long global_buffer_9_glb_topk_inds;
  unsigned long long global_buffer_10_glb_buffer_gather_1;
  unsigned long long global_buffer_11_glb_buffer_gather_2;
  unsigned long long global_buffer_12_glb_buffer_rpn_bbox_permuted;
  unsigned long long global_buffer_13_glb_buffer_nonzero;
  unsigned long long global_buffer_14_result_valid_ind;
  unsigned long long global_buffer_15_glb_buffer_gather_boxes;
  unsigned long long global_buffer_16_glb_buffer_gather_scores;
  unsigned long long global_buffer_17_keep_3nch;
  unsigned long long global_buffer_18_keep_u32_1h;
  unsigned long long global_buffer_19_glb_buffer_boxes;
  unsigned long long global_buffer_20_glb_buffer_scores;
  unsigned long long global_buffer_21_glb_buffer_nms;
  unsigned long long global_buffer_22_gather_mlvl_proposals;
  unsigned long long global_buffer_23_gather_mlvl_scores;
  unsigned long long global_buffer_24_gather_mlvl_ids;
  unsigned long long global_buffer_25_glb_buffer_result_list;
} MaskRCNN_RPN_get_bboxes_global_param_t;

typedef struct MaskRCNN_bbox_pooler_common_spec {
  int mode;
} MaskRCNN_bbox_pooler_common_spec_t;

typedef struct MaskRCNN_bbox_pooler_global_param {
  MaskRCNN_bbox_pooler_common_spec_t spec;
  unsigned long long global_buffer_0_ptr_tmp_res;
  unsigned long long global_buffer_1_ptr_rois_tmp;
} MaskRCNN_bbox_pooler_global_param_t;

typedef struct MaskRCNN_get_bbox_B_common_spec {
  float threshold_score_eq;
  float wh_ratio_log;
  float nms_iou_thr;
  float delta2bbox_means;
  float delta2bbox_stds_0;
  float delta2bbox_stds_1;
} MaskRCNN_get_bbox_B_common_spec_t;

typedef struct MaskRCNN_get_bbox_B_global_param {
  MaskRCNN_get_bbox_B_common_spec_t spec;
  unsigned long long global_buffer_0_means;
  unsigned long long global_buffer_1_stds;
  unsigned long long global_buffer_2_res_bbox;
  unsigned long long global_buffer_3_res_bbox1;
  unsigned long long global_buffer_4_res_bbox0;
  unsigned long long global_buffer_5_res_score0;
  unsigned long long global_buffer_6_res_score1;
  unsigned long long global_buffer_7_res_score2;
  unsigned long long global_buffer_8_res_score3;
  unsigned long long global_buffer_9_res_label2;
  unsigned long long global_buffer_10_result_list;
  unsigned long long global_buffer_11_keep_3nch;
  unsigned long long global_buffer_12_keep_u32_1h;
  unsigned long long global_buffer_13_glb_buffer_boxes;
  unsigned long long global_buffer_14_glb_buffer_scores;
  unsigned long long global_buffer_15_glb_buffer_nms;
  unsigned long long global_buffer_16_glb_buffer_nonzero;
  unsigned long long global_buffer_17_result_valid_ind;
  unsigned long long global_buffer_18_glb_lables;
  unsigned long long global_buffer_19_glb_lables_expand;
} MaskRCNN_get_bbox_B_global_param_t;

typedef struct MaskRCNN_mask_pooler_common_spec {
  int mode;
} MaskRCNN_mask_pooler_common_spec_t;

typedef struct MaskRCNN_mask_pooler_global_param {
  MaskRCNN_mask_pooler_common_spec_t spec;
  unsigned long long global_buffer_0_ptr_rois_buff;
  unsigned long long global_buffer_1_result_filled_det_bboxes;
  unsigned long long global_buffer_2_result_filled_det_labels;
  unsigned long long global_buffer_3_ptr_tmp_res;
  unsigned long long global_buffer_4_ptr_rois_tmp;
} MaskRCNN_mask_pooler_global_param_t;

typedef struct randn_like_spec {
  uint32_t max_shape[MAX_SHAPE_DIMS];
} randn_like_spec_t;
#endif
