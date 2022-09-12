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
struct InferenceParameter {
  std::vector<float *> inputs;
  std::vector<float *> outputs;
  void *handle = nullptr;
};

} // namespace tpu_mlir

/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/InferenceInterface.h.inc"
