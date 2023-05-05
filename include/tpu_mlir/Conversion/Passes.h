//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_CONVERSION_PASSES_H
#define TPU_MLIR_CONVERSION_PASSES_H

#include "tpu_mlir/Conversion/Conversion.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "tpu_mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // TPU_MLIR_CONVERSION_PASSES_H
