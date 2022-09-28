//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"

namespace tpu_mlir {
enum class TypeCastMode {
  DO_NOTHING = 0,
  DO_QUANTIZE = 1,
  DO_DEQUANTIZE = 2,
  DO_CAST = 3,
};

static inline mlir::Type do_nothing(TypeCastMode &mode) {
  mode = TypeCastMode::DO_NOTHING;
  return nullptr;
}

// case: activation input type is the same with output type
mlir::Type type_verify_case_same(mlir::Operation *op, uint64_t opd_idx,
                                 TypeCastMode &mode);

// conv/matmul output if is i32, then input should be quant i8
// else will be the same
mlir::Type type_verify_case_i32(mlir::Operation *op, uint64_t opd_idx,
                                TypeCastMode &mode);

} // namespace tpu_mlir
/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/TypeInterface.h.inc"
