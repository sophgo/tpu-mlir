//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

Value top::GatherOp::lowering_int8_bm1684() {
  llvm_unreachable("GatherOp to be supported");
  return nullptr;
}

Value top::GatherOp::lowering_f32_bm1684() {
  llvm_unreachable("GatherOp to be supported");
  return nullptr;
}
