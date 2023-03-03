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
#include "tpu_mlir/Backend/BM168x/Param.h"

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
typedef void (*sg_set_profile_path)(const char* path);
typedef void (*sg_stas_dump)(void *pid_node);
typedef void (*sg_flops_dump)(long long flops, void *pid_node);
typedef void (*sg_stas_reset)();

namespace tpu_mlir {
namespace backend {

#define CAST_FUNCTION(name) dl_##name = CastToFPtr<name>(#name)
#define CAST_FUNCTION_WITH_SYM(name, sym) dl_##name = CastToFPtr<name>(#sym)

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
  static double getFmtBytes(DATA_TYPE_T data_type);
  static tensor_spec_t value_to_spec(Value v, group_type_t group_type);
  static std::shared_ptr<std::vector<tensor_spec_t>>
  get_input_spec(Operation *op, group_type_t group_type = GROUP_NORMAL);
  static std::shared_ptr<std::vector<tensor_spec_t>>
  get_output_spec(Operation *op, group_type_t group_type = GROUP_NORMAL);
  static std::shared_ptr<std::vector<tensor_spec_t>>
  get_spec(ValueRange values, group_type_t group_type = GROUP_NORMAL);
  static void fix_shape(tensor_spec_t &spec,
                        const std::vector<int64_t> &new_shape);
  static void getBetterNCHW(Value v, int64_t &n, int64_t &c, int64_t &h,
                            int64_t &w);
  static int compare_mode(StringRef mode);
  static int64_t ic_num(double dbytes) { return IC_PARALLEL / dbytes; }
  static stride_4D_t getGlobalStride(int64_t N, int64_t C, int64_t H,
                                     int64_t W);
  static stride_4D_t getLocalStride(int64_t N, int64_t C, int64_t H, int64_t W,
                                    double fmtBytes, bool eu_align = true);
  template <typename T>
  static int64_t dynamic_spec_to_buffer(void *buffer, const T &spec) {
    auto p = static_cast<char *>(buffer);
    memcpy(p, &spec, sizeof(spec));
    p += sizeof(spec);
    return p - static_cast<char *>(buffer);
  }
  static int get_reduce_type(llvm::StringRef mode);

  // -------------------------------------------------------------------
  // global chip config
  // -------------------------------------------------------------------
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
  sg_set_profile_path dl_sg_set_profile_path;
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
