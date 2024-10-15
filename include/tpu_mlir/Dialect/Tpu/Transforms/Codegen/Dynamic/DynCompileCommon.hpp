//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

using namespace tpu_mlir::backend;

/*  Existing Parametric Structure Modification Tips:
 ** 1. New member variables must be added at the
 **    end of the structure.
 ** 2. Every new member variable's size must be
 **    align to 4 bytes.
 */

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
typedef signed char s8;
#define MAX_CONCAT_INPUT_NUM 16
#define MAX_ELET_INPUT_NUM 10
#define MAX_SPLIT_OUTPUT_NUM 8
#define MAX_SHAPE_DIMS 8
#define MAX_YOLO_INPUT_NUM 8
#define MAX_YOLO_ANCHOR_NUM 8
typedef int LayerId;
#ifdef __cplusplus
extern "C" {
#endif

#define IR_TENSOR_TYPE_COEFF 0
#define IR_TENSOR_TYPE_NEURON 1
#define IR_TENSOR_TYPE_SHAPE 2
#define IR_TENSOR_TYPE_ARRAY 3
#define IR_TENSOR_TYPE_FLOW 4

#define TA_MEM_OFFSET (1024 * 1024)

typedef enum fw_layer_type {
  FW_BMNET_CONV = 0,
  FW_BMNET_POOL = 1,
  FW_BMNET_LRN = 2,
  FW_BMNET_FC = 3,
  FW_BMNET_SPLIT = 4,
  FW_BMNET_DATA = 5,
  FW_BMNET_RELU = 6,
  FW_BMNET_CONCAT = 7,
  FW_BMNET_BATCHNORM = 8,
  FW_BMNET_SCALE = 9,
  FW_BMNET_ELTWISE = 10,
  FW_BMNET_PRELU = 11,
  FW_BMNET_LSTM = 12,
  FW_BMNET_PERMUTE = 13,
  FW_BMNET_NORMALIZE = 14,
  FW_BMNET_REORG = 15,
  FW_BMNET_PRIORBOX = 16,
  FW_BMNET_FLATTEN = 17,
  FW_BMNET_RESHAPE = 18,
  FW_BMNET_SOFTMAX = 19,
  FW_BMNET_INTERP = 20,
  FW_BMNET_RPN = 21,
  FW_BMNET_DECONV = 22,
  FW_BMNET_CROP = 23,
  FW_BMNET_MULSHIFT = 24,
  FW_BMNET_POOLTF = 25,
  FW_BMNET_PAD = 26,
  FW_BMNET_ARG = 27,
  FW_BMNET_TILE = 28,
  FW_BMNET_SELECT = 29,
  FW_BMNET_REDUCE = 30,
  FW_BMNET_BROADCAST_BINARY = 31,
  FW_BMNET_CONST_BINARY = 32,
  FW_BMNET_ELTWISE_BINARY = 33,
  FW_BMNET_BIASADD = 34,
  FW_BMNET_ACTIVE = 35,
  FW_BMNET_TENSOR_ARITHMETIC = 36,
  FW_BMNET_SPLIT_TF = 37,
  FW_BMNET_SHAPE_REF = 38,
  FW_BMNET_SHAPE_CONST = 39,
  FW_BMNET_SHAPE_OP = 40,
  FW_BMNET_SHAPE_SLICE = 41,
  FW_BMNET_SHAPE_PACK = 42,
  FW_BMNET_SHAPE_ASSIGN = 43,
  FW_BMNET_SHAPE_REORDER = 44,
  FW_BMNET_EXPAND_DIM = 45,
  FW_BMNET_SQUEEZE_DIM = 46,
  FW_BMNET_REF_PAD = 47,
  FW_BMNET_REF_CROP = 48,
  FW_BMNET_TRANSPOSE = 49,
  FW_BMNET_REDUCE_FULL = 50,
  FW_BMNET_STRIDESLICE = 51,
  FW_BMNET_SPACE2BATCH = 52,
  FW_BMNET_BATCH2SPACE = 53,
  FW_BMNET_OUTPUT = 54,
  FW_BMNET_EXPAND = 55,
  FW_BMNET_EMBEDDING = 56,
  FW_BMNET_TOPK = 57,
  FW_BMNET_CUMSUM = 58,
  FW_BMNET_SHAPE_ADDN = 59,
  FW_BMNET_ROIPOOLING = 60,
  FW_BMNET_CONSTANT_FILL = 61,
  FW_BMNET_SIMPLE_CROP = 62,
  FW_BMNET_SLICELIKE = 63,
  FW_BMNET_ADAPTIVEPOOLING = 64,
  FW_BMNET_BATCH_MATMUL = 65,
  FW_BMNET_UPSAMPLE = 66,
  FW_BMNET_SHAPE_RANGE = 67,
  FW_BMNET_SHAPE_TILE = 68,
  FW_BMNET_SHAPE_REVERSE = 69,
  FW_BMNET_SHAPE_EXPAND_NDIMS = 70,
  FW_BMNET_SHAPE_CAST = 71,
  FW_BMNET_SHAPE_RESHAPE = 72,
  FW_BMNET_SHAPE_REDUCE = 73,
  FW_BMNET_DTYPE_CONVERT = 74,
  FW_BMNET_YOLO = 75,
  FW_BMNET_SSD_DETECT_OUT = 76,
  FW_BMNET_HOST2DEVICE = 77,
  FW_BMNET_DEVICE2HOST = 78,
  FW_BMNET_TENSOR_ARRAY = 79,
  FW_BMNET_TA_WRITE = 80,
  FW_BMNET_TA_READ = 81,
  FW_BMNET_TA_SIZE = 82,
  FW_BMNET_TA_SCATTER = 83,
  FW_BMNET_TA_GATHER = 84,
  FW_BMNET_TA_SPLIT = 85,
  FW_BMNET_TA_CONCAT = 86,
  FW_BMNET_PSROIPOOLING = 87,
  FW_BMNET_SHAPE_UNARY = 88,
  FW_BMNET_SHAPE_SPLIT = 89,
  FW_BMNET_SHAPE_SQUEEZE = 90,
  FW_BMNET_RANK = 91,
  FW_BMNET_WHERE = 92,
  FW_BMNET_YOLOV3_DETECT_OUT = 93,
  FW_BMNET_MASKED_SELECT = 94,
  FW_BMNET_SORT_PER_DIM = 95,
  FW_BMNET_INDEX_SELECT = 96,
  FW_BMNET_NMS = 97,
  FW_BMNET_SLICE = 98,
  FW_BMNET_SHAPE_SIZESLICE = 99,
  FW_BMNET_COEFF2NEURON = 100,
  FW_BMNET_SHAPE_SELECT = 101,
  FW_BMNET_DEPTH2SPACE = 102,
  FW_BMNET_WHERE_SQUEEZE_GATHER = 103,
  FW_BMNET_REVERSE = 104,
  FW_BMNET_BROADCAST_LIKE = 105,
  FW_BMNET_LUT = 106,
  FW_BMNET_MATRIX_BAND_PART = 107,
  FW_BMNET_ARITH_SHIFT = 108,
  FW_BMNET_CONV3D = 109,
  FW_BMNET_POOL3D = 110,
  FW_BMNET_STRIDECALC = 111,
  FW_BMNET_INTERLEAVE = 112,
  FW_BMNET_BITWISE = 113,
  FW_BMNET_BINARY_SHIFT = 114,
  FW_BMNET_GRU = 115,
  FW_BMNET_PYTORCH_LSTM = 116,
  FW_BMNET_MULTI_MASKED_SELECT = 117,
  FW_BMNET_TPU = 118,
  FW_BMNET_SEQUENCE_GEN = 119,
  FW_BMNET_UPSAMPLEMASK = 120,
  FW_BMNET_QUANT_DIV = 121,
  FW_BMNET_GROUP_NORM = 122,
  FW_BMNET_ROI_ALIGN = 123,
  FW_BMNET_EMBEDDING_BAG = 124,
  FW_BMNET_TRIANGULARIZE = 125,
  FW_BMNET_INDEX_PUT = 126,
  FW_BMNET_MASKED_FILL = 127,
  FW_BMNET_GLU = 128,
  FW_BMNET_DEFORM_GATHER = 129,
  FW_BMNET_SCATTERND = 130,
  FW_BMNET_LAYER_NORM = 131,
  FW_BMNET_REQUANT_FP32 = 132,
  FW_BMNET_REQUANT_INT = 133,
  FW_BMNET_DEQUANT_FP32 = 134,
  FW_BMNET_CLIP = 135,
  FW_BMNET_DEQUANT_INT = 136,
  FW_BMNET_SWAP_DIM_INNER = 137,
  FW_BMNET_SWAP_CHANNEL = 138,
  FW_BMNET_SCALE_LUT = 139,
  FW_BMNET_PIXEL_NORM = 140,
  FW_BMNET_NORMAL_SPARSE_CONV3D = 141,
  FW_BMNET_SUBM_SPARSE_CONV3D = 142,
  FW_BMNET_SHAPE_UNSQUEEZE = 143,
  FW_BMNET_UNSQUEEZE = 144,
  FW_BMNET_DECONV3D = 145,
  FW_BMNET_YOLOV5_DETECT_OUT = 146,
  FW_BMNET_ONNX_NMS = 147,
  FW_BMNET_YOLOV8_DETECT_OUT = 148,
  FW_BMNET_SHAPE_ARITH = 149,
  FW_BMNET_RANGE = 150,
  FW_BMNET_RELATIVE_POSITION = 151,
  FW_BMNET_INSTANCENORM = 152,
  FW_BMNET_SCATTERELEMENTS = 153,
  FW_BMNET_RMSNORM = 154,
  FW_BMNET_A16_MATMUL = 155,
  FW_BMNET_FATTENTION = 156,
  FW_BMNET_GATHERND = 157,
  FW_LAYER_GRIDSAMPLER = 158,
  FW_BMNET_GATHERELEMENTS = 159,
  FW_BMNET_SHAPE_CONSTANT_FILL = 160,
  FW_BMNET_MASKRCNNRPNGETBBOXES = 161,
  FW_BMNET_MASKRCNNBBOXPOOLER   = 162,
  FW_BMNET_MASKRCNNGETBBOXB     = 163,
  FW_BMNET_MASKRCNNMASKPOOLER   = 164,
  FW_BMNET_SHAPE_CLIP = 165,
  FW_BMNET_SHAPE_POW = 166,
  FW_BMNET_RANDNLIKE = 167,
  FW_BMNET_SHAPE_TRANSPOSE = 168,
      // global_dynamic step -2: declare FW_BMNET_XXXX
  FW_LAYER_UNKNOWN
} FW_LAYER_TYPE_T;

// There is this data size enum in 1684, which didn't go well. We replace  data
// size enum with our old chum dtype, for good. But we don't want any compatib-
// ility issue. Thus the bit-sharing. Be careful though, you should  *ONLY* set
// dtype for 1684X IR.
#define dtype_and_dsize                                                        \
  union {                                                                      \
    DATA_SIZE_T data_size;                                                     \
    int dtype;                                                                 \
  };

// corresponding DATA_TYPE_T
typedef enum fw_data_size {
  DSIZE_FP32 = 0,
  DSIZE_16 = 1,
  DSIZE_8 = 2,
  DSIZE_INT32 = 3,
} DATA_SIZE_T;

// should has same def in bmruntime_common.h, bmcompiler_common.h
typedef enum fw_data_type {
  FW_DTYPE_FP32 = 0,
  FW_DTYPE_FP16 = 1,
  FW_DTYPE_INT8 = 2,
  FW_DTYPE_UINT8 = 3,
  FW_DTYPE_INT16 = 4,
  FW_DTYPE_UINT16 = 5,
  FW_DTYPE_INT32 = 6,
  FW_DTYPE_UINT32 = 7,
  FW_DTYPE_BFP16 = 8,
} FW_DATA_TYPE_T;

inline static u32 is_float_dtype(FW_DATA_TYPE_T dtype) {
  switch (dtype) {
  case FW_DTYPE_FP32:
  case FW_DTYPE_FP16:
    return 1;
  default:
    return 0;
  }
}

inline static u32 fw_data_size(FW_DATA_TYPE_T dtype) {
  switch (dtype) {
  case FW_DTYPE_FP32:
  case FW_DTYPE_INT32:
  case FW_DTYPE_UINT32:
    return 4;
  case FW_DTYPE_FP16:
  case FW_DTYPE_UINT16:
  case FW_DTYPE_INT16:
    return 2;
  case FW_DTYPE_INT8:
  case FW_DTYPE_UINT8:
    return 1;
  default:
    return 0;
  }
}
typedef struct fw_tensor_arithmetic_layer_param {
  u32 input_n;
  u32 input_c;
  u32 input_h;
  u32 input_w;
  int B_N_is_1;
  int B_H_is_1;
  int B_W_is_1;
  int op;
  int result_add;
  u32 A_is_constant;
  u32 B_is_constant;
  float A_const_val;
  float B_const_val;
  u32 tensor_A_stride_N;
  u32 tensor_A_stride_C;
  u32 tensor_A_stride_H;
  u32 tensor_A_stride_W;
} fw_tensor_arithmetic_layer_param_t;

typedef struct fw_middle_value_param {
  dtype_and_dsize u8 intensor_store_mode;
  u8 outtensor_store_mode;
} fw_middle_value_param_t;

typedef struct fw_mulshift_layer_param {
  u32 ic;
  int mulvalue;
  int mulshiftnum;
  u8 opd0_sign;
  u8 res_sign;
} fw_mulshift_layer_param_t;

typedef struct fw_conv_layer_param {
  u32 ic_oc;
  u32 concat_c; // full channels for local shape
  u32 groups;
  u32 kh_kw;
  u8 dh;
  u8 dw;
  u8 pad_h;
  u8 pad_h_after;
  u8 pad_w;
  u8 pad_w_after;
  u8 stride_h;
  u8 stride_w;
  u8 using_bias;
  u8 if_relu;
  float relu_upper_limit;
  u8 use_winograd;
  u32 c_idx;        // for local concat useage
  u32 reference_id; // for local concat , reference_id is the true ouput tensor
                    // of conv
  u8 rshiftbits;
  u8 opd0_sign;
  u8 opd1_sign;
  u8 opd2_sign;
  u8 res_sign;
  u8 mulshift;
  int mulvalue;
  int mulshiftnum;
  int weight_is_tensor;
  u32 scale_dim[4];
  int scale_axis;
  int scale_axis_num;
  u32 scale_bias;
  u32 if_batchnorm;
  u32 if_scale;
  u32 if_double_buffer;
  u64 weight_global_offset;
  u32 double_buffer_local_offset;
  u32 is_tf_same_pad;
} fw_conv_layer_param_t;

typedef struct fw_rpn_layer_param {
  u32 ic_oc;
  int feat_stride_;
  int base_size_;
  int min_size_;
  int pre_nms_topN_;
  int post_nms_topN_;
  float nms_thresh_;
  float score_thresh_;
  float scale_val_;
  u64 global_offset_1N_buf_;
  u64 arm_reserved_global_offset_;
} fw_rpn_layer_param_t;

typedef struct fw_deconv_layer_param {
  u32 ic_oc;
  u32 groups;
  u32 kh_kw;
  u8 dh;
  u8 dw;
  u8 pad_h;
  u8 pad_h_after;
  u8 pad_w;
  u8 pad_w_after;
  u8 stride_h;
  u8 stride_w;
  u8 using_bias;
  u8 if_relu;
  s8 rshift_num;
  u8 opd0_sign;
  u8 opd1_sign;
  u8 opd2_sign;
  float relu_upper_limit;
  u8 output_padding_h;
  u8 output_padding_w;
  u32 output_dtype; // Note: use DATA_TYPE_T, not DATA_SIZE_T
  u32 imm_buffer_offset;
  u32 using_depthwise;
} fw_deconv_layer_param_t;

typedef struct fw_deconv3d_layer_param {
  int oc;
  int groups;
  int kernel[3]; //kd, kh, kw
  int dilation[3]; //dd, dh, dw
  int pads[6]; //d, d_after, h, h_after, w, w_after
  int stride[3]; //sd, sh, sw
  int using_bias;
  int if_relu;
  float relu_upper_limit;
  int output_padding[3];
  /*for bm1684x*/
  int dtype[4]; //weight, bias, output, kzp
  int kzp_is_const;
  int pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
} fw_deconv3d_layer_param_t;

typedef struct fw_crop_layer_param {
  u32 shape_tensor_id;
  u32 bottom_n;
  u32 bottom_c;
  u32 top_n;
  u32 top_c;
  u32 top_h;
  u32 top_w;
  int offset_n;
  int offset_c;
  int offset_h;
  int offset_w;
} fw_crop_layer_param_t;

typedef struct fw_reshape_layer_param {
  u8 new_dims;
  int new_shape[MAX_SHAPE_DIMS];
  int bottom_n;
  int bottom_c;
  u32 bottom_tensor_id;
  u64 global_buffer_addr;
} fw_reshape_layer_param_t;

typedef struct fw_concat_loc_layer_param {
  u32 c;
  u32 h;
  u32 w;
  u32 version;
  u8 concat_axis;
} fw_concat_loc_layer_param_t;

typedef struct fw_fc_layer_param {
  u32 input_neuron_num;
  u32 output_neuron_num;
  u8 transpose;
  u8 using_bias;
  u8 if_activated;
  int active_type;
  u8 channel_shared;
  float shared_slope;
  s8 rshift_num;
  u8 opd0_sign;
  u8 opd1_sign;
  u8 opd2_sign;
  u8 res_sign;
  float relu_upper_limit;
  int out_dims;
  u8 if_use_scale;
  u8 if_asymmetric;
  u8 if_bias_float;
  u8 if_perchannel;
  float scale;
  short weight_offset;
  short output_offset;
  int weight_is_datatensor;
  u8 version;
  u8 res_16b;
  u8 output_sign;
  float perlayer_bias;
} fw_fc_layer_param_t;

typedef struct fw_pool_layer_param {
  u32 ic;
  u32 kh_kw;
  u8 pad_h_top;
  u8 pad_h_bottom;
  u8 pad_w_left;
  u8 pad_w_right;
  u8 stride_h;
  u8 stride_w;
  u8 is_avg_pool;
  u8 avg_pooling_mode;
  u8 if_relu;
  float relu_upper_limit;
  u8 opd0_sign;
  u8 res_sign;
  u8 is_global_pool;
  u8 out_ceil_mode;
} fw_pool_layer_param_t;

typedef struct fw_pooltf_layer_param {
  u32 ic;
  u32 kh;
  u32 kw;
  u8 pad_h_top;
  u8 pad_h_bottom;
  u8 pad_w_left;
  u8 pad_w_right;
  u8 stride_h;
  u8 stride_w;
  u8 is_avg_pool;
  u8 if_relu;
  u8 opd0_sign;
  u8 res_sign;
  u8 is_global_pool;
  float relu_upper_limit;
} fw_pooltf_layer_param_t;

typedef struct fw_split_tf_layer_param {
  int axis;
  int split_size[MAX_SPLIT_OUTPUT_NUM];
  int split_num;
  u64 buffer_addr;
  int input_num; // =2 means split_size is dynamic
} fw_split_tf_layer_param_t;

typedef struct fw_lrn_layer_param {
  u32 ic;
  float alpha;
  float beta;
  float k;
  u8 size;
  u8 opd0_sign;
  float scale_in;
  float scale_out;
} fw_lrn_layer_param_t;

typedef struct fw_eltwise_layer_param {
  u32 ic;
  u8 op_code;
  u8 in_num;
  u8 if_relu;
  u8 opd_sign[MAX_ELET_INPUT_NUM];
  s8 in_rshift_num[MAX_ELET_INPUT_NUM];
  float relu_upper_limit;
} fw_eltwise_layer_param_t;

typedef struct fw_prelu_layer_param {
  u32 ic;
  float shared_slope;
  u8 channel_shared;
  u8 rshift_bit;
  float relu_upper_limit;
  u8 in_sign;
  u8 out_sign;
  u8 slope_sign;
} fw_prelu_layer_param_t;

typedef struct fw_normalize_layer_param {
  u32 ic;
  float eps;
  float scale_val;
  u8 across_spatial;
  u8 channel_shared;
  u8 if_relu;
  float relu_upper_limit;
  u8 in_sign;
  u8 out_sign;
} fw_normalize_layer_param_t;

typedef struct fw_softmax_layer_param {
  // u32 inner_num;
  u32 softmax_dim;
  float scale_val;
  // u32 out_dim;
  u64 global_offset_1N_buf;
  u8 log;
} fw_softmax_layer_param_t;

typedef struct fw_batchnorm_layer_param {
  u32 ic;
  float scale_ma;
  float eps;
  u8 opd0_sign;
  u8 opd1_sign;
  u8 opd2_sign;
  u8 if_relu;
  float relu_upper_limit;
  u32 scale_dim[4];
  int scale_axis;
  int scale_axis_num;
  u32 if_scale;
  u32 scale_bias;
} fw_batchnorm_layer_param_t;

typedef struct fw_scale_layer_param {
  u32 ic;
  int shape_axis;
  int shape_axis_num;
  u8 using_bias;
  u8 if_relu;
  float relu_upper_limit;
  u8 opd0_sign;
  u8 opd1_sign;
  u8 opd2_sign;
  int scale_is_neuron;
  int bias_is_neuron;
} fw_scale_layer_param_t;

typedef struct fw_pad_layer_param {
  u32 ic;
  float pad_val;
  int pad_mode;
  int paddings[4][2];
  u64 input_global_offset_1N_buf;
  u64 output_global_offset_1N_buf;
  u8 is_dynamic;
} fw_pad_layer_param_t;

typedef struct fw_arg_layer_param {
  u32 ic;
  int axis;
  int method;
  int input_sign;
  u64 input_glb_buf;
  u64 imm_index_glb_buf;
  short is_index_int32;
  short select_last_index;
} fw_arg_layer_param_t;

typedef struct fw_active_layer_param {
  u32 ic;
  int active_type;
  u8 if_relu;
  float relu_upper_limit;
  float input_scale_back2float;
  float output_scale_back2float;
  u8 opd_sign;
  u8 res_sign;
} fw_active_layer_param_t;

typedef struct fw_permute_layer_param {
  int permute_order[4];
} fw_permute_layer_param_t;

typedef struct {
  u8 coeff_is_fixed;
  u8 input_is_coeff;
  u8 type;
  u8 input_dims;
  int input_shape[MAX_SHAPE_DIMS];
  int tile_coeff[MAX_SHAPE_DIMS];
  u64 buffer_addr;
  int dtype;
} fw_tile_layer_param_t;

typedef struct {
  u8 index_dim;
  u8 index_is_coeff;
  u8 lut_is_coeff;
  u8 stmode_dtype; // bottom_store_mode:   0: 4N,      1: 1N;
                   // top_data_dtype:      0: float,   1: int8 or uint8;
                   // top_store_mode:      0: 1N,      1: 4N;
                   // stmode_dtype = bottom_store_mode + top_data_dtype << 2 +
                   // top_store_mode << 4;
  int index_shape[MAX_SHAPE_DIMS];
  int dtype;
} fw_lut_layer_param_t;

typedef struct {
  int lower;
  int upper;
} fw_matrix_band_part_layer_param_t;

typedef struct {
  int dims;
  union {
    int data[MAX_SHAPE_DIMS];
    int shape[MAX_SHAPE_DIMS];
  };
  int is_in_host;
  int dtype;
} fw_shape_const_layer_param_t;

typedef struct {
  int num_inputs;
  int axis;
} fw_shape_pack_layer_param_t;

typedef struct {
  int binary_op;
} fw_shape_op_layer_param_t;

typedef struct {
  int begin : 8;
  int end : 8;
  int step : 8;
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
} fw_shape_slice_layer_param_t;

typedef struct {
  u8 order[MAX_SHAPE_DIMS];
} fw_shape_reorder_layer_param_t;

typedef struct {
  u8 axis_list[MAX_SHAPE_DIMS];
  int axis_num;
} fw_squeeze_layer_param_t;

typedef struct {
  int axis;
  int ndims;
  int is_coeff;
  int input_shape[MAX_SHAPE_DIMS];
  int input_dims;
} fw_expand_dims_layer_param_t;

typedef struct {
  u8 order[MAX_SHAPE_DIMS];
  u64 buffer_addr;
  u8 order_is_dynamic;
} fw_transpose_layer_param_t;

typedef struct {
  int pad_mode;
  float pad_val;
  u64 global_buffer_addr;
} fw_ref_pad_layer_param_t;

typedef struct {
  float s0_value;
  float s1_value;
  u8 s0_is_const;
  u8 s1_is_const;
  u8 if_relu;
  float relu_upper_limit;
  u8 scalea;
  u8 nshifta;
  u8 scaleb;
  u8 nshiftb;
  u8 in_sign;
  u8 s0_sign;
  u8 s1_sign;
  u8 reserved; // 4 bytes align
} fw_select_layer_param_t;

typedef struct {
  int dims;
  u64 buffer_addr;
} fw_where_layer_param_t;

typedef struct {
  int input_dtype_bytes;
  int input_store_mode;
  int mask_dtype_bytes;
  int mask_store_mode;
  u64 buffer_addr;
  u64 buffer_addr_ex;
  u8 bcast_from_begin;
  u8 reserve[3];
} fw_masked_select_layer_param_t;

typedef struct {
  int input_dtype_bytes[MAX_SHAPE_DIMS];
  int input_store_mode[MAX_SHAPE_DIMS];
  int mask_dtype_bytes;
  int mask_store_mode;
  u64 buffer_addr;
  u64 buffer_addr_ex;
  u8 bcast_from_begin;
  int multi_num;
} fw_multi_masked_select_layer_param_t;

typedef struct {
  int input_dtype_bytes[8];
  int input_store_mode[8];
  int mask_dtype_bytes;
  int mask_store_mode;
  u64 buffer_addr;
  u64 buffer_addr_ex;
  u8 bcast_from_begin;
  int input_num;
  int axes[8];
} fw_where_squeeze_gather_layer_param_t;

typedef struct {
  int dim;
  u64 buffer_addr;
  int index_is_coeff;
  int index_num;
} fw_index_select_layer_param_t;

typedef struct {
  float iou_threshold;
  float score_threshold;
  int input_num;
  u64 buffer_addr;
} fw_nms_layer_param_t;

typedef struct {
  int dim;
  int is_argsort;
  int stable;
  int descending;
  u64 buffer_addr;
} fw_sort_per_dim_layer_param_t;

typedef struct {
  int reduce_method;
  int reduce_w;
} fw_reduce_layer_param_t;

typedef struct {
  u8 a_dims;
  u8 a_is_coeff;
  u8 b_dims;
  u8 b_is_coeff;
  int binary_op;
  u8 if_relu;
  int a_shape[MAX_SHAPE_DIMS];
  int b_shape[MAX_SHAPE_DIMS];
  float relu_upper_limit;
  u64 buffer_addr;
  u8 scale[2];
  s8 rshift_num[2];
  u8 opd_sign[3];
} fw_broadcast_binary_layer_param_t;

typedef struct {
  float b_value;
  int binary_op;
  u8 inversed;
  u8 if_relu;
  float relu_upper_limit;
  u8 scale[2];
  s8 rshift_num[2];
  u8 opd_sign[2];
  u8 reserve[2];
} fw_const_binary_layer_param_t;

typedef struct {
  int binary_op;
  u8 a_is_coeff;
  u8 b_is_coeff;
  u8 if_relu;
  float relu_upper_limit;
  u8 scale[2];
  s8 rshift_num[2];
  u8 opd_sign[2];
  u8 reserve[2];
} fw_eltwise_binary_layer_param_t;

typedef struct {
  u8 if_relu;
  float relu_upper_limit;
  int in_rshift_num;
  int bottom_coeff;
  u8 opd0_sign;
  u8 opd1_sign;
} fw_biasadd_layer_param_t;

typedef struct {
  short keep_dims;
  short axis_num;
  int reduce_method;
  u8 axis_list[MAX_SHAPE_DIMS];
  u64 buffer_addr;
  int input_sign;
  float input_scale;
  float output_scale;
} fw_reduce_full_layer_param_t;

typedef struct {
  u8 is_coeff;
  u8 st_way;
  int concat_size;
} fw_concat_input_info_t;

typedef struct {
  u8 input_num;
  u8 base_dims;
  u8 concat_axis;
  int base_shape[MAX_SHAPE_DIMS];
} fw_concat_layer_param_t;

typedef struct {
  int begin_mask;
  int end_mask;
  int begin_index[MAX_SHAPE_DIMS];
  int end_index[MAX_SHAPE_DIMS];
  int stride[MAX_SHAPE_DIMS];
  int shape_size;
  int shrink_axis_mask;
  int new_axis_mask;
  int ellipsis_mask;
  u64 buffer_global_addr;
  u64 imm_global_addr;
  u8 is_dynamic;
} fw_stride_slice_layer_param_t;

typedef struct {
  u8 block_is_dynamic;
  u8 pad_is_dynamic;
  int block_sizes[2];
  int pad_sizes[4];
  u64 buffer_addr;
} fw_space2batch_layer_param_t;

typedef struct {
  u8 block_is_dynamic;
  u8 crop_is_dynamic;
  int block_sizes[2];
  int crop_sizes[4];
  u64 buffer_addr;
} fw_batch2space_layer_param_t;

typedef struct {
  u8 shape_is_fixed;
  u8 dims;
  u8 platform_sp;
  int shape[MAX_SHAPE_DIMS];
  int pad_bag;
  int pad_end;
  int opd0_sign;
  int align_corners;
  int half_pixel_centers;
} fw_interp_layer_param_t;

typedef struct {
  u8 input_is_coeff;
  u8 input_dims;
  int input_shape[MAX_SHAPE_DIMS];
  int output_shape_is_fixed;
  int output_shape[MAX_SHAPE_DIMS];
  int output_dim;
  u64 buffer_addr;
} fw_expand_layer_param_t;

typedef struct {
  int k;
  u8 dim;
  u8 is_dynamic;
  u8 descending;
  u64 buffer_addr;
} fw_topk_layer_param_t;

typedef struct {
  u8 dim;
} fw_cumsum_layer_param_t;

typedef struct {
  int input_num;
} fw_shape_addn_layer_param_t;

typedef struct {
  u32 value;
  int type_len;
  int dtype;
} fw_constant_fill_layer_param_t;

typedef struct fw_simple_crop_layer_param {
  int crop_sizes[8];
} fw_simple_crop_layer_param_t;

typedef struct {
  int axis[MAX_SHAPE_DIMS];
  int axis_num;
  int input_is_coeff;
  int input_dims;
  int input_shape[MAX_SHAPE_DIMS];
} fw_slicelike_layer_param_t;

typedef struct {
  u64 buffer_addr; // for fix8b
  u32 slice_mask;
} fw_slice_layer_param_t;

typedef struct {
  u32 slice_mask;
} fw_shape_sizeslice_layer_param_t;

typedef struct {
  int pooled_h;
  int pooled_w;
} fw_adaptive_pool_layer_param_t;

typedef struct {
  int in0_is_coeff;
  int in0_shape[MAX_SHAPE_DIMS];
  int in0_dims;
  int in1_is_coeff;
  int in1_shape[MAX_SHAPE_DIMS];
  int in1_dims;
  int if_relu;
  float relu_upper_limit;
} fw_batch_matmul_layer_param_t;

typedef struct {
  int tile_coeff[MAX_SHAPE_DIMS];
  int is_dynamic;
} fw_shape_tile_layer_param_t;

typedef struct {
  int axis;
} fw_shape_reverse_layer_param_t;

typedef struct {
  int axis;
  int ndims;
} fw_shape_expand_ndims_layer_param_t;

typedef struct {
  int dst_type;
  int output_is_shape;
} fw_shape_cast_layer_param_t;

typedef struct {
  int keep_dims;
  int reduce_method;
  int axis_num;
  int axis_list[MAX_SHAPE_DIMS];
} fw_shape_reduce_layer_param_t;

typedef struct {
  int src_type;
  int dst_type;
  int src_stmode;
  int dst_stmode;
  int round_mode;
} fw_dtype_convert_layer_param_t;

typedef struct {
  int axis;
  int split_size[MAX_SPLIT_OUTPUT_NUM];
  int split_num;
} fw_shape_split_layer_param_t;

typedef struct {
  int method;
} fw_shape_unary_layer_param_t;

typedef struct {
  u8 axis_list[MAX_SHAPE_DIMS];
  int axis_num;
} fw_shape_squeeze_layer_param_t;

typedef struct {
  int need_sync;
  int input_is_coeff;
  int input_dims;
  int input_shape[MAX_SHAPE_DIMS];
  int input_dtype;
} fw_tensor_array_op_param_t;

typedef struct {
  int input_is_shape;
} fw_shape_ref_layer_param_t;

typedef struct {
  int input_is_shape;
} fw_rank_layer_param_t;

typedef struct {
  int shape[MAX_SHAPE_DIMS];
  int dims;
  int dtype;
} fw_coeff2neuron_layer_param_t;

typedef struct {
  float s0_value;
  float s1_value;
  u8 s0_is_const;
  u8 s1_is_const;
} fw_shape_select_layer_param_t;

typedef struct {
  u64 buffer_addr;
  int block_sizes[2];
  u8 in_is_nchw;
  u8 is_inversed;
  u8 out_is_nchw : 1;
  // Share bits for compatibility concerns.
  // When new lib encounters a old model,
  // the sharing bits of out_is_nchw will all be zeros.
  // So it falls back to CRD mode.
  u8 is_crd : 1;
} fw_depth2space_layer_param_t;

typedef struct {
  int axes_num;
  char lhs_axes[MAX_SHAPE_DIMS];
  char rhs_axes[MAX_SHAPE_DIMS];
  u64 buffer_addr;
} fw_broadcast_like_layer_param_t;

typedef struct {
  int ic;
  int oc;
  int groups;
  int kt;
  int kh;
  int kw;
  int dt;
  int dh;
  int dw;
  int pad_t;
  int pad_t_after;
  int pad_h;
  int pad_h_after;
  int pad_w;
  int pad_w_after;
  int stride_t;
  int stride_h;
  int stride_w;
  int method;
  int scale_axis;
  int scale_axis_num;
  u8 using_bias;
  u8 with_eltwise_add;
  u8 if_relu;
  float relu_upper_limit;
  // fix8b-specific parameters
  u8 input_sign;
  u8 output_sign;
  u8 weight_sign;
  u8 bias_sign;
  int rshift_num;
} fw_conv3d_layer_param_t;

typedef struct {
  int bias;
  int bidirection;
  int batch_first;
  int num_layers;
  u64 buffer_addr;
} fw_gru_layer_param_t, fw_pytorch_lstm_layer_param_t;

typedef struct {
  int kt;
  int kh;
  int kw;
  int pad_t;
  int pad_t_after;
  int pad_h;
  int pad_h_after;
  int pad_w;
  int pad_w_after;
  int stride_t;
  int stride_h;
  int stride_w;
  u8 is_avg_pool;
  u8 avg_pooling_mode;
  u8 is_global_pool;
  u8 out_ceil_mode;
  u8 if_relu;
  float relu_upper_limit;
  u64 buffer_addr;
  // fix8b-api-related parameters
  int using_bias;
  int rshift_num;
  int rshift_type;
  u8 opd0_sign;
  u8 opd1_sign;
  u8 opd2_sign;
  u8 res0_sign;
  u32 local_buffer_offset;
} fw_pool3d_layer_param_t;

typedef struct {
  int input_num;
  int output_num;
  int param_size;
  u64 buffer_addr;
  u64 buffer_size;
  int op_type;
} fw_tpu_layer_param_t;
// global_dynamic step -1: declare layer_param struct

typedef struct {
  int input_sign;
  int shift_type;
  int shift_num;
  int shift_mode;
  int shift_is_const;
  int is_num_neuron;
  int b_shape[MAX_SHAPE_DIMS];
  FW_DATA_TYPE_T b_dtype;
  int ROUND_RSHIFT_OUTDTYPE;
} fw_arith_shift_layer_param_t;

typedef struct {
  int pooled_h;
  int pooled_w;
  float spatial_scale;
  int roi_nums;
  int opd0_sign;
} fw_roi_pooling_layer_param_t;

typedef struct {
  int output_dim;
  int group_size;
  float spatial_scale;
  int roi_nums;
  int opd0_sign;
} fw_psroi_pooling_layer_param_t;

typedef struct {
  int size;
  int if_relu;
  int opd0_sign;
} fw_upsample_layer_param_t;

typedef struct {
  int dims;
  int shape[MAX_SHAPE_DIMS];
  float min_sizes[MAX_SHAPE_DIMS];
  int real_min_size;
  float max_sizes[MAX_SHAPE_DIMS];
  int real_max_size;
  float aspect_ratios[MAX_SHAPE_DIMS];
  int real_aspect_size;
  float variance[MAX_SHAPE_DIMS];
  int real_variance_size;
  int num_priors;
  int img_w;
  int img_h;
  float step_w;
  float step_h;
  float offset;
  float thTop;
  int bottom_0_width;
  int bottom_0_height;
  int bottom_1_width;
  int bottom_1_height;
  int dim;
  u8 has_dim;
  u8 flip;
  u8 clip;
  int version;
} fw_priorbox_layer_param_t;

typedef struct fw_yolo_layer_param {
  int n;
  int classes;
  int coords;
  int background;
  int softmax;
} fw_yolo_layer_param_t;

typedef struct fw_yolov3_detect_out_layer_param {
  int input_num;
  int num_classes;
  int num_boxes;
  int mask_group_size;
  int keep_top_k;
  float confidence_threshold;
  float nms_threshold;
  float bias[18];
  float anchor_scale[3];
  float mask[9];
  int yolo_box_flag; // 0: yolov3_detect_out, 1:paddle_yolo_box
  int clip_bbox;     // used for paddle yolo_box 1:true, 0:false
  float scale;       // used for paddle yolo_box
} fw_yolov3_detect_out_layer_param_t;

typedef struct fw_yolov5_detect_out_layer_param {
    int keep_top_k;
    int agnostic_nms;
    int max_hw;
    float nms_threshold;
    float confidence_threshold;
} fw_yolov5_detect_out_layer_param_t;

typedef struct fw_yolov5_decode_detect_out_layer_param {
    int input_num;
    int batch_num;
    int num_classes;
    int num_boxes;
    int keep_top_k;
    float nms_threshold;
    float confidence_threshold;
    float anchors[2 * MAX_YOLO_INPUT_NUM * MAX_YOLO_ANCHOR_NUM];
    float anchor_scale[MAX_YOLO_ANCHOR_NUM];
    int agnostic_nms;
} fw_yolov5_decode_detect_out_layer_param_t;

typedef struct fw_yolov8_detect_out_layer_param {
    int keep_top_k;
    int agnostic_nms;
    int max_hw;
    float nms_threshold;
    float confidence_threshold;
} fw_yolov8_detect_out_layer_param_t;

typedef struct {
  int num_classes;
  int share_location;
  int background_label_id;
  int code_type;
  int variance_encoded_in_target;
  int keep_top_k;
  float confidence_threshold;
  float nms_threshold;
  float eta;
  int top_k;
  int onnx_nms; // 1:onnx_nms
} fw_ssd_detect_out_layer_param_t;

typedef struct fw_reorg_layer_param {
  int stride;
  int reverse;
  int out_split_c;
  int out_merge_c;
  u64 buffer_addr;
} fw_reorg_layer_param_t;

typedef struct fw_tensorarray_layer_param {
  FW_DATA_TYPE_T dtype;
  int elem_shape[MAX_SHAPE_DIMS];
  int elem_dims;
  int clear_after_read;
  int dynamic_size;
  int elem_identical;
  int global_buffer_size;
} fw_tensorarray_layer_param_t;

typedef struct {
  int axis;
} fw_reverse_layer_param_t;

typedef struct fw_lstm_layer_param {
  int input_n;
  int seq_len;
  int input_dim;
  int input_static_dim;
  int output_dim;
  int user_define_cont;
  int with_input_static;
  int expose_hidden;
  u64 buffer_addr;
  u64 buffer_addr_ex;
} fw_lstm_layer_param_t;

typedef struct fw_stridecalc_layer_param {
  int op_code;
  int input_info; // a_is_coeff, b_is_coeff, a_is_const, b_is_const
  float a_const_val;
  float b_const_val;
  int offset[MAX_SHAPE_DIMS];
  int stride[MAX_SHAPE_DIMS];
  int a_shape[MAX_SHAPE_DIMS];
  int b_shape[MAX_SHAPE_DIMS];
  u8 if_relu;
  u8 opd0_dtype;
  u8 opd1_dtype;
  u8 res_dtype;
  int b_stride[MAX_SHAPE_DIMS];
} fw_stridecalc_layer_param_t;

typedef struct fw_binary_shift_param {
  u8 b_is_const;
  u8 a_is_coeff;
  u8 b_is_coeff;
  s8 rshift_num;
  u8 opd0_dtype;
  u8 opd1_dtype;
  u8 res_dtype;
  u8 inversed;
  int binary_op;
  int b_const_val;
  int a_shape[MAX_SHAPE_DIMS];
  int b_shape[MAX_SHAPE_DIMS];
  u32 buffer_offset;
} fw_binary_shift_layer_param_t;

typedef struct fw_interleave_layer_param {
  u8 axis;
  u8 step;
  u8 a_is_coeff;
  u8 b_is_coeff;
} fw_interleave_layer_param_t;

typedef struct fw_bitwise_layer_param {
  int if_const;
  int b_value;
  int bitwise_op;
} fw_bitwise_layer_param_t;

typedef struct fw_upsample_mask_layer_param {
  int size;
} fw_upsample_mask_layer_param_t;

typedef struct fw_group_norm_layer_param {
  int group_num;
  int affine; // 0: no weight and bias, 1: weight, 2: bias, 3: both
  float eps;
  u32 local_buffer_addr;
} fw_group_norm_layer_param_t;

typedef enum tensor_gdma_type {
  LD_INPUT_NEURON = 0,
  ST_OUTPUT_NEURON = 1,
  LD_ITM_NEURON = 2,
  ST_ITM_NEURON = 3,
  LD_COEFF = 4,
  LD_COEFF_NERUON = 5,
  LD_COEFF_WINOGRAD = 6,
  MV_ITM_NEURON = 7,
  MV_OUTPUT_NEURON = 8,
  MV_ITM_EXTEND_NEURON = 9,
  ST_ITM_EXTEND_NEURON = 10,
  LD_G2L2 = 11,
  ST_OUTPUT_EXTEND_NEURON = 12,
  LD_ITM_EXTEND_NEURON = 13,
  GDMA_TYPE_NUMBER
} tensor_gdma_type_t;

typedef struct fw_gdma_ld_in_neuron {
  u32 local_offset;
  u32 ic_and_tensor_id;
  u32 c_idx_and_reference_id; // for local concat use, reference_id is the true
                              // tensor id
  u32 concat_c; // full c for tensor_id, to_ic if store_mode is STORE_3IC
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N, 3: STORE_3IC
  u32 consumer_num;
} fw_gdma_ld_in_neuron_t;

typedef struct fw_gdma_ld_g2l2 {
  u64 global_offset;
  u32 l2_offset;
  u32 tensor_id;
  u8 tensor_type; // 0: input neuron 1: itm neuron 2: coeff
  u32 length;
  dtype_and_dsize
} fw_gdma_ld_g2l2_t;

typedef struct fw_gdma_st_out_neuron {
  u32 local_offset;
  u32 ic_and_tensor_id;
  u32 concat_c_and_cidx;
  u32 concat_tensor_id;
  u8 merge_npu_c;
  u8 split_c_num;
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
} fw_gdma_st_out_neuron_t;

typedef struct fw_gdma_ld_itm_neuron {
  u64 global_offset;
  u32 local_offset;
  u32 ic_and_tensor_id;
  u32 c_idx_and_reference_id; // for local concat use, reference_id is the true
                              // tensor id
  u32 concat_c;               // full c for tensor_id
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
  u32 consumer_num;
} fw_gdma_ld_itm_neuron_t;

typedef struct fw_gdma_st_itm_neuron {
  u64 global_offset;
  u32 local_offset;
  u32 ic_and_tensor_id;
  u32 concat_c_and_cidx;
  u32 concat_tensor_id;
  u8 merge_npu_c;
  u8 split_c_num;
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
} fw_gdma_st_itm_neuron_t;

typedef struct fw_gdma_mv_itm_neuron {
  u64 src_global_offset;
  u64 dst_global_offset;
  u32 ic_and_tensor_id;
  u32 concat_c_and_cidx;
  u32 concat_tensor_id;
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
} fw_gdma_mv_itm_neuron_t;

typedef struct fw_gdma_mv_out_neuron {
  u64 src_global_offset;
  u32 ic_and_tensor_id;
  u32 concat_c_and_cidx;
  u32 concat_tensor_id;
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
} fw_gdma_mv_out_neuron_t;

typedef struct fw_gdma_coeff {
  u64 global_offset;
  u32 local_offset;
  u32 ic_oc;
  u32 kh_kw;
  u8 one_time;
  u8 winograd_coeff;
  u8 groups;
  dtype_and_dsize u8 if_double_buffer;
  u32 depth;
  FW_LAYER_TYPE_T layer_type;
} fw_gdma_coeff_t;

typedef struct fw_gdma_coeff_neuron {
  u64 global_offset;
  u32 local_offset;
  u32 c_and_w;
  u32 pad_h_top_and_h_step;
  u32 h_slice_and_h;
  u8 version_and_one_time;
  u8 n_is_one;
  dtype_and_dsize u32 n;
  u32 tensor_id;
  u8 store_mode;
  u32 consumer_num;
} fw_gdma_coeff_neuron_t;

typedef struct fw_input_tensor_info {
  u32 tensor_id_and_max_hslice;
  u32 stride_h_and_kh;
  u32 pad_h_top_and_bottom;
  u32 min_pool_kh;
} fw_input_tensor_info_t;

typedef struct fw_gdma_ld_itm_extend_neuron {
  u64 global_offset;
  u32 local_offset;
  u32 ic;
  u32 tensor_id;
  u32 c_idx;
  u32 reference_id; // for local concat use, reference_id is the true tensor
  u32 concat_c;     // full c for tensor_id, to_ic if store_mode is STORE_3IC
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N, 3: STORE_3IC
  u32 consumer_num;
} fw_gdma_ld_itm_extend_neuron_t;

typedef struct fw_gdma_st_itm_extend_neuron {
  u64 global_offset;
  u32 local_offset;
  u32 ic;
  u32 tensor_id;
  u32 concat_c;
  u32 cidx;
  u32 concat_tensor_id;
  dtype_and_dsize u8 merge_npu_c;
  u8 split_c_num;
  u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
} fw_gdma_st_itm_extend_neuron_t;

typedef struct fw_gdma_mv_itm_extend_neuron {
  u64 src_global_offset;
  u64 dst_global_offset;
  u32 tensor_id;
  u32 cidx;
  u32 concat_tensor_id;
  dtype_and_dsize u32 ic;
  u32 concat_c;
  u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
} fw_gdma_mv_itm_extend_neuron_t;

typedef struct fw_gdma_st_out_extend_neuron {
  u32 local_offset;
  u32 ic;
  u32 tensor_id;
  u32 concat_c;
  u32 cidx;
  u32 concat_tensor_id;
  u8 merge_npu_c;
  u8 split_c_num;
  dtype_and_dsize u8 store_mode; // 0: 1N, 1: 2N, 2: 4N
} fw_gdma_st_out_extend_neuron_t;

typedef struct fw_timestep_base_info {
  u32 ts_num_and_split_tensor_num;
  u8 max_nslice_deprecated; // for compatiable to old dynamic ir
  u8 input_tensor_num;
  u8 output_tensor_num;
  u8 flags; // bit0: is_h_split, bit1:using max_nslice, bit2-bit4: group_type,
            // bit5: consumer_opt
  u8 swpipl_stage_num;
  u32 max_nslice;
} fw_timestep_base_info_t;

// used only for nodechip_dynamic_fullnet_compile
typedef struct {
  u64 addr;
  int dims;
  int shape[MAX_SHAPE_DIMS];
  dtype_and_dsize int elem_num;
} fw_fullnet_tensor_info_t;

typedef struct {
  int padding_idx;
} fw_embedding_layer_param_t;

// typedef struct {
//   int num_embeddings;
//   int embedding_dim;
//   int mode;
// } fw_embedding_bag_layer_param_t;
// use to return dynamic output tensor info
typedef struct {
  int dims;
  int shape[MAX_SHAPE_DIMS];
  int elem_num;
} fw_dynamic_output_info_t;

// must be same as TENSOR_TYPE_T in bmcompiler_net_param.h
// must be same as include/tpu_mlir/Backend/BM168x/Param.h
typedef enum {
  FW_BMNET_NEURON = 0,
  FW_BMNET_COEFF = 1,
  FW_BMNET_COEFF_NEURON = 2,
  FW_BMNET_COEFF_FC = 3,
  FW_BMNET_COEFF_WINOGRAD = 4,
  FW_BMNET_NEURON_FC = 5,
  FW_BMNET_NEURON_CONST = 6,
  FW_BMNET_NEURON_SHAPE = 7,
  FW_BMNET_NEURON_CPU = 8,
  FW_BMNET_NEURON_ARRAY = 9,
  FW_BMNET_NEURON_FLOW = 10,
  FW_BMNET_NEURON_3IC = 11,
  FW_TENSOR_UNKNOWN,
} FW_TENSOR_TYPE_T;

static inline int fw_is_COEFF(FW_TENSOR_TYPE_T tensor_type) {
  return (tensor_type == FW_BMNET_COEFF ||
          tensor_type == FW_BMNET_COEFF_NEURON ||
          tensor_type == FW_BMNET_COEFF_FC ||
          tensor_type == FW_BMNET_COEFF_WINOGRAD);
}

static inline int fw_is_NEURON(FW_TENSOR_TYPE_T tensor_type) {
  return (tensor_type == FW_BMNET_NEURON || tensor_type == FW_BMNET_NEURON_FC ||
          tensor_type == FW_BMNET_NEURON_CONST ||
          tensor_type == FW_BMNET_NEURON_CPU ||
          tensor_type == FW_BMNET_NEURON_3IC);
}

// used for tpu layer io info parsing
typedef struct {
  u8 ttype;
  u8 dtype;
  u8 store_mode;
  u8 reserved;
} fw_tpu_tensor_type_info_t;

typedef struct {
  u64 addr;                        // can be local_offset or global_addr
  u32 dims;                        // dimension of tensor
  int elem_num;                    // real element number
  int shape[MAX_SHAPE_DIMS];       // for global or local calculate
  int slice_pad[MAX_SHAPE_DIMS];   // for local calculate
  int slice_shape[MAX_SHAPE_DIMS]; // for local calculate
  int processed[MAX_SHAPE_DIMS];   // for local calculate
  int *host_data;                  // used for read or write host data
  u8 ttype;
  u8 dtype;
  u8 store_mode;
} fw_tpu_tensor_t;

typedef int (*tpu_global_entry_t)(
    const int op_type, const void *param, const int param_size,
    const fw_tpu_tensor_t *in_tensors, const int in_num, const u64 buffer_addr,
    const u64 buffer_size, fw_tpu_tensor_t *out_tensors, const int out_num);
typedef int (*tpu_local_entry_t)(const int op_type, const void *param,
                                 const int param_size,
                                 const fw_tpu_tensor_t *in_tensors,
                                 const int in_num, const u64 buffer_offset,
                                 const u64 buffer_size,
                                 fw_tpu_tensor_t *out_tensors,
                                 const int out_num);
typedef int (*tpu_shape_entry_t)(const int op_type, const void *param,
                                 const int param_size,
                                 const fw_tpu_tensor_t *in_tensors,
                                 const int in_num, fw_tpu_tensor_t *out_tensors,
                                 const int out_num);

static inline void dtype_to_dsize_and_sign(int dtype, DATA_SIZE_T *dsize,
                                           int *sign) {
  if (dsize) {
    if (FW_DTYPE_INT8 == dtype || FW_DTYPE_UINT8 == dtype) {
      *dsize = DSIZE_8;
    } else if (FW_DTYPE_UINT16 == dtype || FW_DTYPE_INT16 == dtype ||
               FW_DTYPE_FP16 == dtype) {
      *dsize = DSIZE_16;
    } else if (FW_DTYPE_INT32 == dtype || FW_DTYPE_UINT32 == dtype) {
      *dsize = DSIZE_INT32;
    } else if (FW_DTYPE_FP32 == dtype) {
      *dsize = DSIZE_FP32;
    }
  }
  if (sign) {
    *sign = (FW_DTYPE_INT8 == dtype || FW_DTYPE_INT16 == dtype ||
             FW_DTYPE_INT32 == dtype);
  }
}

static inline DATA_SIZE_T get_dynamic_compiler_tensor_datasize(Value v) {
  DATA_SIZE_T data_size;
  auto data_type = BM168x::getDataType(v);
  switch (data_type) {
  case DTYPE_INT8:
  case DTYPE_UINT8:
    data_size = DSIZE_8;
    break;
  case DTYPE_INT16:
  case DTYPE_UINT16:
  case DTYPE_FP16:
    data_size = DSIZE_16;
    break;
  case DTYPE_INT32:
  case DTYPE_UINT32:
    data_size = DSIZE_INT32;
    break;
  default:
    data_size = DSIZE_FP32;
    break;
  }
  return data_size;
}

#ifdef __cplusplus
}
#endif
