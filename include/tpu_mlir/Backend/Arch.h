//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include <assert.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace tpu_mlir {
namespace backend {

class Arch {
public:
  static void init();
  static int64_t NPU_NUM;
  static int64_t EU_BYTES;
  static int64_t LMEM_BYTES;
  static int64_t LMEM_BANKS;
  static int64_t LMEM_BANK_BYTES;
  static llvm::StringRef LIB_NAME;
  static bool ALIGN_4N;
  static module::Chip chip;
  // dbytes is 0.5 for INT4
  static int64_t eu_num(double dbytes);
  static int64_t get_n_align(int64_t dtype_bytes) {
    return ALIGN_4N ? (4 / dtype_bytes) : 1;
  }
  static size_t get_gmem_bytes(Value v);
  static int64_t get_tensor_lmem_bytes(Value v, int64_t n, int64_t c, int64_t h,
                                       int64_t w, bool eu_align = true);
  static int64_t get_tensor_lmem_bytes(Value v, int64_t slice_n,
                                       int64_t slice_h, bool eu_align = true);
  static int64_t get_weight_lmem_bytes(Value v, bool eu_align = true);

  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName) {
    assert(DL.isValid());
    auto fPtr = DL.getAddressOfSymbol(symbolName);
    if (fPtr == nullptr) {
      llvm::errs() << "can't find symbol: " << symbolName << "\n";
      llvm_unreachable(symbolName);
    }
    return reinterpret_cast<FPtrTy>(fPtr);
  }

protected:
  static Arch *inst;
  llvm::sys::DynamicLibrary DL;
  Arch(){};
  virtual ~Arch() = 0;
  void load_library();
};
} // namespace backend
} // namespace tpu_mlir
