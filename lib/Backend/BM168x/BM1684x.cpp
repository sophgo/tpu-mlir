//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;
using namespace mlir;

uint32_t BM1684x::get_bdc_len(int bdc_num, int group_id) {
  if (bdc_num == 0) {
    return 0;
  }
  assert(group_id < bdc_bytes.size());
  return bdc_bytes[group_id];
}

uint32_t BM1684x::get_gdma_len(int gdma_num, int group_id) {
  if (gdma_num == 0) {
    return 0;
  }
  assert(group_id < gdma_bytes.size());
  return gdma_bytes[group_id];
}

template <typename FPtrTy> FPtrTy BM1684x::CastToFPtr(const char *symbolName) {
  assert(DL.isValid());
  auto fPtr = DL.getAddressOfSymbol(symbolName);
  if (fPtr == nullptr) {
    llvm::errs() << "can't find symbol: " << symbolName << "\n";
    llvm_unreachable(symbolName);
  }
  return reinterpret_cast<FPtrTy>(fPtr);
}

#define CAST_FUNCTION(name) dl_##name = CastToFPtr<name>(#name)

void BM1684x::load_functions() {
  BM168x::load_functions();
  CAST_FUNCTION(cmd_id_divide);
  CAST_FUNCTION(cmd_id_merge);
  CAST_FUNCTION(load_lookup_tables);
  CAST_FUNCTION(store_cmd_end);
  CAST_FUNCTION(set_cmd_len_ptr);
}

void BM1684x::init() {
  BM168x::init();
  dl_load_lookup_tables();
  dl_set_cmd_len_ptr((void *)&gdma_bytes, (void *)&bdc_bytes);
}

void BM1684x::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  dl_store_cmd_end();
}
