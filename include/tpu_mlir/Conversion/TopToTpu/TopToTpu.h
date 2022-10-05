//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_CONVERSION_TOPTOTPU_H
#define TPU_MLIR_CONVERSION_TOPTOTPU_H

#include "tpu_mlir/Conversion/TopToTpu/TopLowering.h"

namespace mlir {
#define GEN_PASS_DECL
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

std::unique_ptr<Pass> createConvertTopToTpu();

} // namespace tpu_mlir

#endif // TPU_MLIR_CONVERSION_TOPTOTPU_H
