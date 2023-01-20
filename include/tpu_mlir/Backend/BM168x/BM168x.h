//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/Arch.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cmd_id_node;
typedef struct cmd_id_node CMD_ID_NODE;

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
} STORE_MODE_T;

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
  ELTWISE_PRODUCT = 0,
  ELTWISE_ADD = 1,
  ELTWISE_MAX = 2,
} ELTWISE_OPCODE_T;

typedef struct bmcompiler_mem_info {
  uint64_t addr;
  uint64_t size;
  uint64_t offset;
} bm_mem_desc_t;
typedef struct bmcompiler_mem_info bm_device_mem_t;

#define MAX_SHAPE_DIMS 8
typedef struct local_tensor_spec {
  uint64_t addr;
  int32_t dtype;
  int32_t dims;
  int32_t shape[MAX_SHAPE_DIMS];
  uint8_t consume_num;
  int *host_data;
  int elem_num;
} tensor_spec_t;

typedef enum {
  /* 3D group if this group has CONV3D/DECONV3D/POOL3D
   * for 1684 float32, data in local memory storage as {d * n, c, h, w}
   * for 1684 int8, data in local memory storage as {n, d * c, h, w}
   * for 1684X, data in local memory storage as {d * n, c, h, w}
   * data in global memory always storage as {n, c, d, h, w}
   * group_type < 8, because 1684 dynamic compile reserved `3bit` for group_type
   */
  GROUP_NORMAL = 0,
  GROUP_3D = 1,
} group_type_t;

typedef struct stride {
  int64_t N, C, H, W;
} stride_4D_t;

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

typedef struct bcbinary_common_spec {
  int32_t binary_type;
  int32_t if_relu;
  float relu_upper_limit;
  int32_t scale_A;
  int32_t scale_B;
  int32_t rshift_A;
  int32_t rshift_B;
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
  unsigned long long input_addr;
  unsigned long long slope_addr;
  unsigned long long output_addr;
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

typedef struct {
  unsigned long long input_addr;
  unsigned long long weight_addr;
  unsigned long long bias_addr;
  unsigned long long output_addr;
  unsigned long long mean_addr;
  unsigned long long rstd_addr;
  int shape[MAX_SHAPE_DIMS];
  int dims;
  int axis;
  float eps;
  int affine; // 0: no weight and bias, 1: weight, 2: bias, 3: both
  int need_mean;
  int need_rstd;
  int dtype;
} layer_norm_global_param_t;

typedef struct {
  unsigned int input_addr;
  unsigned int weight_addr;
  unsigned int bias_addr;
  unsigned int output_addr;
  unsigned int mean_addr;
  unsigned int rstd_addr;
  unsigned int buffer_addr;
  int input_n;
  int input_c;
  int input_h;
  int input_w;
  int depth;
  float eps;
  int affine;
  int need_mean;
  int need_rstd;
  int dtype;
} layer_norm_local_param_t;

typedef struct reshape_spec {
    int32_t dims;
    int32_t shape[MAX_SHAPE_DIMS];
} reshape_spec_t ;
#ifdef __cplusplus
}
#endif

typedef int (*cmodel_init)(int node_idx, uint64_t global_mem_size);
typedef void (*cmodel_deinit)(int node_idx);
typedef void *(*create_cmd_id_node)();
typedef void (*destroy_cmd_id_node)(void *pid_node);
typedef void (*set_cmd_id_cycle)(void *pid_node, int val);
typedef int (*get_cmd_id_cycle)(void *pid_node);
typedef void (*reset_cmd_id)(void *pid_node);
typedef void (*allow_store_cmd)();
typedef void (*forbid_store_cmd)();
typedef void (*use_atomic_cmodel)();
typedef void (*forbid_atomic_cmodel)();
typedef void *(*get_global_memaddr)(int node_idx);
typedef void (*set_cmd_buffer_ptr)(void *gdma_buffer_ptr, void *bdc_buffer_ptr);
typedef void (*set_cmd_id_prefix)(void *pid_node, const char *name_prefix);
typedef void (*allow_atomic_cmodel_assert)();
typedef void (*forbid_atomic_cmodel_assert)();

typedef void (*tensor_stride_move_gen_cmd)(
    int local_mem_start_addr, int local_mem_idx, uint64_t sys_mem_start_addr,
    int src_N, int src_C, int src_H, int src_W, uint32_t src_N_stride,
    uint32_t src_C_stride, uint32_t src_H_stride, uint32_t src_W_stride,
    uint32_t dst_N_stride, uint32_t dst_C_stride, uint32_t dst_H_stride,
    uint32_t dst_W_stride, int src_format, int direction, int transpose,
    CMD_ID_NODE *pid_node);

typedef void (*tensor_compact_move_gen_cmd)(
    int local_mem_start_addr, int local_mem_idx, uint64_t sys_mem_start_addr,
    int src_N, int src_C, int src_H, int src_W, int src_format, int direction,
    int transpose, CMD_ID_NODE *pid_node);

typedef void (*tensor_broadcast_move_gen_cmd)(
    uint64_t src_addr, int src_local_idx, int dst_lmem_start_addr,
    int dst_local_idx, int src_N, int src_H, int src_W, int dst_C,
    uint32_t src_N_stride, uint32_t src_H_stride, uint32_t dst_N_stride,
    uint32_t dst_H_stride, int data_format, int stride_enable, int direction,
    CMD_ID_NODE *pid_node);

typedef void (*set_total_id_ptr)(uint32_t *gdma_total_id_ptr,
                                 uint32_t *bdc_total_id_ptr, void *cmdid_node,
                                 void *gdma_group_id_ptr,
                                 void *bdc_group_id_ptr, int *cmdid_groupnum);
typedef void (*cmd_id_divide)(void *p_cmd_src, void *p_cmd_dst0,
                              void *p_cmd_dst1);
typedef void (*cmd_id_merge)(void *p_cmd_dst, void *p_cmd_src0,
                             void *p_cmd_src1);
typedef void (*sg_set_profile_dump)(bool enable);
typedef void (*sg_stas_dump)(void *pid_node);
typedef void (*sg_flops_dump)(long long flops, void *pid_node);
typedef void (*sg_stas_reset)();

namespace tpu_mlir {
namespace backend {

#define CAST_FUNCTION(name)                                                    \
  dl_##name = CastToFPtr<name>(#name)
#define CAST_FUNCTION_WITH_SYM(name, sym)                                      \
  dl_##name = CastToFPtr<name>(#sym)

class BM168x : public Arch {

public:
  static BM168x *instance() { return (BM168x *)inst; }
  // -------------------------------------------------------------------
  // helper functions for global
  // -------------------------------------------------------------------
  static void call_global_func(const char *symbolName, void *params,
                               int param_size);
  static void call_local_func(const char *symbolName, void *params,
                              int param_size);
  static void call_global_func(const char *symbolName, void *params,
                               int param_size, void *input, void *output);
  static void call_local_func(const char *symbolName, void *params,
                              int param_size, void *info, void *input,
                              void *output);
  static DATA_TYPE_T getDataType(Type type);
  static DATA_TYPE_T getDataType(Value v);
  static int getGdmaFormat(DATA_TYPE_T data_type);
  static int getFmtBytes(DATA_TYPE_T data_type);
  static tensor_spec_t value_to_spec(Value v);
  static std::shared_ptr<std::vector<tensor_spec_t>>
  get_input_spec(Operation *op);
  static std::shared_ptr<std::vector<tensor_spec_t>>
  get_output_spec(Operation *op);
  static std::shared_ptr<std::vector<tensor_spec_t>>
  get_spec(ValueRange values);
  static void fix_shape(tensor_spec_t &spec,
                        const std::vector<int32_t> &new_shape);
  static int compare_mode(StringRef mode);
  static int64_t ic_num(double dbytes) { return IC_PARALLEL / dbytes; }
  static stride_4D_t getGlobalStride(int64_t N, int64_t C, int64_t H,
                                     int64_t W);
  static stride_4D_t getLocalStride(int64_t N, int64_t C, int64_t H, int64_t W,
                                    int fmtBytes, bool eu_align = true);
  static int64_t ALIGNMENT;
  static int64_t IC_PARALLEL;
  static uint64_t GMEM_START_ADDR;
  static uint64_t CTX_START_ADDR;
  static const uint64_t CMODEL_GMEM_SIZE = 0x100000000ull;
  // GDMA Format
  static int GDMA_VALUE_FORMAT_UINT8;
  static int GDMA_VALUE_FORMAT_INT8;
  static int GDMA_VALUE_FORMAT_FLOAT16;
  static int GDMA_VALUE_FORMAT_FLOAT32;
  static int GDMA_VALUE_FORMAT_INT16;
  static int GDMA_VALUE_FORMAT_INT32;
  static int GDMA_VALUE_FORMAT_BFLOAT16;
  static int GDMA_VALUE_FORMAT_INT4;
  static int GDMA_VALUE_FORMAT_NUM;

  // -------------------------------------------------------------------
  // functions from nodechip
  // -------------------------------------------------------------------
  cmodel_init dl_cmodel_init;
  cmodel_deinit dl_cmodel_deinit;
  create_cmd_id_node dl_create_cmd_id_node;
  destroy_cmd_id_node dl_destroy_cmd_id_node;
  set_cmd_id_cycle dl_set_cmd_id_cycle;
  get_cmd_id_cycle dl_get_cmd_id_cycle;
  reset_cmd_id dl_reset_cmd_id;
  allow_store_cmd dl_allow_store_cmd;
  forbid_store_cmd dl_forbid_store_cmd;
  use_atomic_cmodel dl_use_atomic_cmodel;
  forbid_atomic_cmodel dl_forbid_atomic_cmodel;
  get_global_memaddr dl_get_global_memaddr;
  set_cmd_buffer_ptr dl_set_cmd_buffer_ptr;
  set_cmd_id_prefix dl_set_cmd_id_prefix;
  allow_atomic_cmodel_assert dl_allow_atomic_cmodel_assert;
  forbid_atomic_cmodel_assert dl_forbid_atomic_cmodel_assert;
  tensor_stride_move_gen_cmd dl_tensor_stride_move_gen_cmd;
  tensor_compact_move_gen_cmd dl_tensor_compact_move_gen_cmd;
  tensor_broadcast_move_gen_cmd dl_tensor_broadcast_move_gen_cmd;
  set_total_id_ptr dl_set_total_id_ptr;
  cmd_id_divide dl_cmd_id_divide;
  cmd_id_merge dl_cmd_id_merge;
  sg_set_profile_dump dl_sg_set_profile_dump;
  sg_stas_dump dl_sg_stas_dump;
  sg_flops_dump dl_sg_flops_dump;
  sg_stas_reset dl_sg_stas_reset;

public:
  // -------------------------------------------------------------------
  // functions for codegen
  // -------------------------------------------------------------------
  virtual void before_codegen();
  virtual void after_codegen(int64_t flops = 0);

  void *get_gmem_addr(uint64_t addr);
  void *get_gmem_addr(const bm_device_mem_t &mem);
  void bm_memcpy_s2d(const bm_device_mem_t &dst, void *src);
  void bm_memcpy_d2s(void *dst, const bm_device_mem_t &src);
  void value_s2d(Value v, void *src);
  void value_d2s(Value v, void *dst);
  void divide_sync_id();
  void merge_sync_id();

  // arch info
  virtual uint32_t get_bdc_len(int bdc_num, int group_id) = 0;
  virtual uint32_t get_gdma_len(int gdma_num, int group_id) = 0;

  void set_command_issue_flag(bool value);
  void reset_cmd_id_node();
  int64_t get_gdma_cycle();
  int64_t get_bdc_cycle();
  int64_t get_cmd_cycle();

  void reset_command_flag() {
    dl_use_atomic_cmodel();
    dl_allow_atomic_cmodel_assert();
    dl_forbid_store_cmd();
  }

public:
  std::vector<uint32_t> bdc_buffer;
  std::vector<uint32_t> gdma_buffer;
  uint32_t gdma_total_id;
  uint32_t bdc_total_id;
  std::vector<uint32_t> gdma_group_id;
  std::vector<uint32_t> bdc_group_id;
  std::vector<uint32_t> gdma_bytes;
  std::vector<uint32_t> bdc_bytes;
  int cmdid_groupnum;
  void *cmdid_node;
  void *bdc_node;
  void *gdma_node;

protected:
  BM168x(){};
  virtual ~BM168x() = 0;
  virtual void load_functions();
  virtual void start_env();
  virtual void end_env();

protected:
  static BM168x *bm168x;
  bool really_issue_command;
};

} // namespace backend
} // namespace tpu_mlir
