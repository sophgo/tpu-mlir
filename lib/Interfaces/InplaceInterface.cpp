//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/InplaceInterface.h"

namespace tpu_mlir {

int64_t getInplaceResultIndex(mlir::Operation *op, int64_t opd_index) {
  auto inplace_op = dyn_cast<InplaceInterface>(op);
  if (!inplace_op) {
    return -1;
  }
  return inplace_op.get_inplace_result_index(opd_index);
}

int64_t getInplaceOperandIndex(mlir::Operation *op, int64_t result_index) {
  auto inplace_op = dyn_cast<InplaceInterface>(op);
  if (!inplace_op) {
    return -1;
  }
  return inplace_op.get_inplace_operand_index(result_index);
}

} // namespace tpu_mlir
