//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Helper/Module.h"

namespace tpu_mlir {
namespace backend {

class BM1686 : public BM1684X {
public:
  static BM1686 &instance() {
    static BM1686 BM1686;
    return BM1686;
  }

protected:
  BM1686() {
    NPU_NUM = 32;
    EU_BYTES = 16;
    LMEM_BYTES = 1 << 17; // 128KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 32;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    LIB_NAME = "libbackend_1686.so";
  };
  virtual ~BM1686(){};
};
} // namespace backend
} // namespace tpu_mlir
