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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::ReciprocalOp::init(InferenceParameter &p) {
  return success();
}

void tpu::ReciprocalOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ReciprocalOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  float const_s = const_val().convertToDouble();
  if (out_type.isa<FloatType>()) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = const_s / p.inputs[0][i];
    }
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else {
    llvm_unreachable("Not Implemented");
  }
  return success();
}
