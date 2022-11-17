//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Module.h"

namespace tpu_mlir {
namespace backend {
using namespace tpu_mlir::helper;

class Athena2 : public BM1684x {
public:
  static Athena2 &instance() {
    static Athena2 inst;
    return  inst;
  }
public:
  virtual uint64_t get_ctx_start_addr() override{
    return BM1684x::get_ctx_start_addr();}
  virtual int64_t get_npu_num() override { return NPU_NUM; }
  virtual int64_t get_eu_bytes() override { return EU_BYTES; }
  virtual int64_t get_lmem_bytes() override { return LMEM_BYTES; }
  virtual int64_t get_lmem_banks() override { return LMEM_BANKS; }
  virtual uint32_t get_bdc_len(int bdc_num, int group_id) override {
    return BM1684x::get_bdc_len(bdc_num, group_id);}
  virtual uint32_t get_gdma_len(int gdma_num, int group_id) override {
    return BM1684x::get_gdma_len(gdma_num, group_id);}

  static const int64_t NPU_NUM = 32;
  static const int64_t EU_BYTES = 16;
  static const int64_t LMEM_BYTES = 1 << 17; // 128KB
  static const int64_t LMEM_BANKS = 16;
  static const int64_t LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
  static constexpr llvm::StringRef LIB_NAME = "libbackend_athena2.so";

protected:
  Athena2(){chip = Module::Chip::ATHENA2;};
  ~Athena2(){};
  virtual const char *get_lib_name() override { return LIB_NAME.data(); };
};
} // namespace backend
} // namespace tpu_mlir
