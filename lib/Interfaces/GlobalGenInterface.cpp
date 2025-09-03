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
  if (module::isOpInBlock(op)) {
    return false;
  }
  if (module::getCoreNum() < 2) {
    return false;
  }
  if (!op->hasAttrOfType<BoolAttr>("multicore")) {
    return false;
  }

  return op->getAttrOfType<BoolAttr>("multicore").getValue() == true;
}

} // namespace tpu_mlir
