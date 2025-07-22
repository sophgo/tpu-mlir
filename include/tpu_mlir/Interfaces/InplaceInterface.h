//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {

int64_t getInplaceResultIndex(mlir::Operation *op, int64_t opd_index);

int64_t getInplaceOperandIndex(mlir::Operation *op, int64_t result_index);

} // namespace tpu_mlir

#include "tpu_mlir/Interfaces/InplaceInterface.h.inc"
