//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"



LogicalResult tpu::ReciprocalOp::init(InferenceParameter &p) {
  return success();
}

void tpu::ReciprocalOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ReciprocalOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  float const_s = getConstVal().convertToDouble();
  if (out_type.isa<FloatType>()) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = const_s / p.inputs[0][i];
    }
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else {
    llvm_unreachable("Not Implemented");
  }
  return success();
}
