//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV184X.h"

using namespace tpu_mlir::backend;

void CV184X::load_functions() { BM1684X::load_functions(); }

void CV184X::before_codegen() { BM1684X::before_codegen(); }

void CV184X::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  dl_store_cmd_end();
}
