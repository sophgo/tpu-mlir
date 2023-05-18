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
#include "tpu_mlir/Support/Module.h"

typedef void (*set_tiu_freq)(float freq);

namespace tpu_mlir {
namespace backend {
template<typename type>
class BM1686 : public BM1684X {
public:
  static BM1686 &instance() {
    static BM1686<type> BM1686;
    return BM1686;
  }
  set_tiu_freq dl_set_tiu_freq;

private:
  void set_simulation_freq(void) {
    CAST_FUNCTION(set_tiu_freq);
    if(get_frequance() == 0) {
      dl_set_tiu_freq(static_cast<float>(type::value));
    } else {
      dl_set_tiu_freq(static_cast<float>(get_frequance()));
    }
  }
protected:
  BM1686() {
    NPU_NUM = 32;
    EU_BYTES = 16;
    LMEM_BYTES = 1 << 17; // 128KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 32;
    ALIGNMENT = 0x1000;
    GMEM_START_ADDR = 0x100000000ull;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    CTX_START_ADDR = GMEM_START_ADDR;
    LIB_BACKEND_NAME = "libbackend_1686.so";
    // GDMA format
    GDMA_VALUE_FORMAT_INT8 = 0;
    GDMA_VALUE_FORMAT_FLOAT16 = 1;
    GDMA_VALUE_FORMAT_FLOAT32 = 2;
    GDMA_VALUE_FORMAT_INT16 = 3;
    GDMA_VALUE_FORMAT_INT32 = 4;
    GDMA_VALUE_FORMAT_BFLOAT16 = 5;
    GDMA_VALUE_FORMAT_INT4 = 6;
    GDMA_VALUE_FORMAT_NUM = 7;

    start_env();
    set_simulation_freq();
  };
  virtual ~BM1686(){};
};
} // namespace backend
} // namespace tpu_mlir
