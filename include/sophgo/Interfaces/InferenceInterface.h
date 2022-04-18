#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "omp.h"

//#define DEBUG_TPU_INFER

namespace sophgo {
struct InferenceParameter {
  std::vector<float *> inputs;
  std::vector<float *> outputs;
  void *handle = nullptr;
};

// commond interface
int omp_schedule(int count);

void relu(float *src, float *dst, int64_t size, mlir::Type elem_type = nullptr);

} // namespace sophgo

/// Include the ODS generated interface header files.
#include "sophgo/Interfaces/InferenceInterface.h.inc"

