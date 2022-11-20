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

class Athena2 : public BM1684x {
public:
  static Athena2 &instance() {
    static Athena2 athena2;
    return athena2;
  }

protected:
  Athena2() {
    NPU_NUM = 32;
    EU_BYTES = 16;
    LMEM_BYTES = 1 << 17; // 128KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 32;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    LIB_NAME = "libbackend_athena2.so";
  };
  virtual ~Athena2(){};
};
} // namespace backend
} // namespace tpu_mlir
