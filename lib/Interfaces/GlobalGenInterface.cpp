//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/GlobalGenInterface.h"
#include "tpu_mlir/Interfaces/GlobalGenInterface.cpp.inc"

using namespace mlir;

namespace tpu_mlir {

bool supportMultiCore(mlir::Operation *op) {
  auto gl = dyn_cast<GlobalGenInterface>(op);
  if (!gl) {
    return false;
  }
  if (module::isOpInBlock(op)) {
    return false;
  }
  if (module::getCoreNum() < 2) {
    return false;
  }
  return gl.support_multi_core();
}

} // namespace tpu_mlir
