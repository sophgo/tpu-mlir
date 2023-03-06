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

// only one output, and output shape is the same with the first input shape
void common_shape_inference(mlir::Operation *op);

} // namespace tpu_mlir
/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/ShapeInterface.h.inc"
