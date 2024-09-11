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

typedef void (*load_lookup_tables)();
typedef void (*store_cmd_end)();
typedef void (*set_cmd_len_ptr)(void *gdma_cmd_len_ptr, void *bdc_cmd_len_ptr);
typedef void (*enable_active_mode)(bool enable);
typedef void (*set_ts_fe_cmd_id_ptr)(void *gdma_cmdid_node, void *bdc_cmdid_node);
// tpu-kernel
typedef void (*tpu_set_id_node)(void *node);
typedef void (*tpu_get_id_node)(void *node);
typedef void (*set_id_node)(void *cmdid_node);

// multi-core switch interface from backend
typedef void (*backend_api_set_core_info)(int, int);
typedef unsigned int (*backend_api_get_tpu_inst_size)(const char *);
typedef const unsigned char *(*backend_api_get_tpu_inst_data)(const char *);
typedef unsigned int (*backend_api_get_tpu_inst_group_number)();
typedef const unsigned int *(*backend_api_get_tpu_inst_number_per_group)(
    const char *);
typedef const unsigned int *(*backend_api_get_tpu_inst_size_per_group)(
    const char *);
typedef void (*backend_api_clear_tpu_inst_data)();
typedef unsigned int (*backend_api_get_total_id)(const char *);

namespace tpu_mlir {
namespace backend {
class BM1684X : public BM168x {
public:
  static BM1684X &instance() {
    static BM1684X bm1684x;
    return bm1684x;
  }

public:
  // -------------------------------------------------------------------
  // functions from nodechip
  // -------------------------------------------------------------------
  // clang-format off
  load_lookup_tables dl_load_lookup_tables;
  store_cmd_end dl_store_cmd_end;
  enable_active_mode dl_enable_active_mode;
  set_cmd_len_ptr dl_set_cmd_len_ptr;
  set_ts_fe_cmd_id_ptr dl_set_ts_fe_cmd_id_ptr;

  tpu_set_id_node dl_tpu_set_id_node;
  tpu_get_id_node dl_tpu_get_id_node;

  backend_api_set_core_info dl_backend_api_set_core_info;
  backend_api_get_tpu_inst_size dl_backend_api_get_tpu_inst_size;
  backend_api_get_tpu_inst_data dl_backend_api_get_tpu_inst_data;
  backend_api_get_tpu_inst_group_number dl_backend_api_get_tpu_inst_group_number;
  backend_api_get_tpu_inst_number_per_group dl_backend_api_get_tpu_inst_number_per_group;
  backend_api_get_tpu_inst_size_per_group dl_backend_api_get_tpu_inst_size_per_group;
  backend_api_clear_tpu_inst_data dl_backend_api_clear_tpu_inst_data;
  backend_api_get_total_id dl_backend_api_get_total_id;
  // clang-format on
public:
  virtual void after_codegen(int64_t flops = 0) override;
  // arch info
  virtual uint32_t get_bdc_len(int bdc_num, int group_id) override;
  virtual uint32_t get_gdma_len(int gdma_num, int group_id) override;
  virtual unsigned int get_total_id(const char *engine_name) override {
    return dl_backend_api_get_total_id(engine_name);
  }
  virtual unsigned int get_inst_number_per_group(const char *engine_name,
                                                 int group_idx) override {
    const unsigned int *inst_ptr =
        dl_backend_api_get_tpu_inst_number_per_group(engine_name);
    if (inst_ptr)
      return inst_ptr[group_idx];
    else
      return 0;
  }
  virtual unsigned int get_group_number() override {
    return dl_backend_api_get_tpu_inst_group_number();
  }
  virtual const unsigned char *get_inst_data(const char *engine_name) override {
    return dl_backend_api_get_tpu_inst_data(engine_name);
  }
  virtual unsigned int get_inst_size(const char *engine_name) override {
    return dl_backend_api_get_tpu_inst_size(engine_name);
  };

public:
  // specific global info
  static constexpr llvm::StringRef LIB_KERNEL_NAME =
      "libbm1684x_kernel_module.so";

protected:
  BM1684X() : BM168x(TypeID::get<BM1684X>()) {
    if (chip != module::Chip::BM1684X) {
      return;
    }
    code = std::make_unique<BM168x::Code>();
    NPU_NUM = 64;
    EU_BYTES = 64;
    LMEM_BYTES = 1 << 18; // 256KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 64;
    ALIGNMENT = 0x1000;
    GMEM_START_ADDR = 0x100000000ull;
    L2_SRAM_START_ADDR = 0x10000000ull;
    L2_SRAM_SIZE = 0x1FB000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    CTX_START_ADDR = GMEM_START_ADDR;
    COEFF_START_ADDR = GMEM_START_ADDR;
    // GDMA format
    GDMA_VALUE_FORMAT_INT8 = 0;
    GDMA_VALUE_FORMAT_FLOAT16 = 1;
    GDMA_VALUE_FORMAT_FLOAT32 = 2;
    GDMA_VALUE_FORMAT_INT16 = 3;
    GDMA_VALUE_FORMAT_INT32 = 4;
    GDMA_VALUE_FORMAT_BFLOAT16 = 5;
    GDMA_VALUE_FORMAT_INT4 = 6;
    GDMA_VALUE_FORMAT_NUM = 7;
    LIB_BACKEND_NAME = "libbackend_1684x.so";
    core_num = module::getCoreNum();
    start_env();
    load_custom_functions();
    dl_enable_active_mode(true);
    dl_set_id_node(code->cmdid_node);
    dl_set_ts_fe_cmd_id_ptr(code->gdma_node, code->bdc_node);
  };
  virtual ~BM1684X() { end_env(); };

  virtual void start_env() override;
  virtual void load_functions() override;
  virtual void before_codegen() override;
  int core_num;

private:
  set_id_node dl_set_id_node;
  void load_custom_functions();
};
} // namespace backend
} // namespace tpu_mlir
