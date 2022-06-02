//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "omp.h"

//#define DEBUG_TPU_INFER

namespace tpu_mlir {
struct InferenceParameter {
  std::vector<float *> inputs;
  std::vector<float *> outputs;
  void *handle = nullptr;
};

// commond interface
int omp_schedule(int count);

void relu(float *src, float *dst, int64_t size, mlir::Type elem_type = nullptr);

} // namespace tpu_mlir

/// Include the ODS generated interface header files.
#include "tpu_mlir/Interfaces/InferenceInterface.h.inc"
