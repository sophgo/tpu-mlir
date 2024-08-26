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
/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/GlobalGenInterface.h.inc"
#include "tpu_mlir/Interfaces/DynGlobalGenInterface.h.inc"

namespace tpu_mlir {

bool supportMultiCore(mlir::Operation *op);

} // namespace tpu_mlir
