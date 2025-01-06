//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/SGTPUV8.h"

using namespace tpu_mlir::backend;

void SGTPUV8::load_functions() { BM1684X::load_functions(); }

void SGTPUV8::before_codegen() { BM168x::before_codegen(); }

void SGTPUV8::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  dl_store_cmd_end();
}
