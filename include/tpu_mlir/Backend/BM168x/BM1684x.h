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

namespace tpu_mlir {
namespace backend {
class BM1684x : public BM168x {
public:
  static BM1684x &instance() {
    static BM1684x inst;
    return inst;
  }

public:
  // -------------------------------------------------------------------
  // functions from nodechip
  // -------------------------------------------------------------------
  load_lookup_tables dl_load_lookup_tables;
  store_cmd_end dl_store_cmd_end;
  set_cmd_len_ptr dl_set_cmd_len_ptr;

public:
  virtual void init() override;
  virtual void after_codegen(int64_t flops = 0) override;

  // arch info
  virtual uint32_t get_bdc_len(int bdc_num, int group_id) override;
  virtual uint32_t get_gdma_len(int gdma_num, int group_id) override;

protected:
  BM1684x() {
    chip = Module::Chip::BM1684x;
    NPU_NUM = 64;
    EU_BYTES = 64;
    LMEM_BYTES = 1 << 18; // 256KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 64;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    CTX_START_ADDR = GMEM_START_ADDR;
    LIB_NAME = "libbackend_1684x.so";
  };
  ~BM1684x(){};

  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName);
  virtual void load_functions() override;
};
} // namespace backend
} // namespace tpu_mlir
