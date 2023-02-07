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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::AddConstOp::init(InferenceParameter &p) { return success(); }

void tpu::AddConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::AddConstOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
  if (module::isUniformQuantized(getOutput())) {
    auto i_qtype = module::getUniformQuantizedType(getInput());
    double izp = 0;
    if (module::isUniformQuantized(getInput()))
      izp = i_qtype.getZeroPoint();
    auto out_qtype = module::getUniformQuantizedType(getOutput());
    double ozp = 0;
    auto out_type = module::getStorageType(getOutput());
    if (asym == true)
      ozp = out_qtype.getZeroPoint();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      double sum = p.inputs[0][i] - izp;
      sum = applyMultiplierAndRShift(sum, getMultiplier(), 0);
      sum += getConstVal().convertToDouble();
      sum = applyMultiplierAndRShift(sum, 1, getRshift());

      if (getDoRelu() && sum < 0) {
        sum = 0;
      }
      p.outputs[0][i] = saturate(sum, out_type);
    }
  }
  else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = p.inputs[0][i] + getConstVal().convertToDouble();
    }
    if (getDoRelu()) {
      auto limit = getReluLimit().convertToDouble();
      function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
    }
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  }
  return success();
}

LogicalResult tpu::AddConstOp::canonicalize(AddConstOp op,
                                            PatternRewriter &rewriter) {
  if (std::abs(op.getConstVal().convertToDouble()) < 1e-7) {
    op.getResult().replaceAllUsesWith(op.getInput());
    return success();
  }
  return failure();
};
