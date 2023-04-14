//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::GatherOp::init(InferenceParameter &p) { return success(); }
void tpu::GatherOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GatherOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *inds = p.inputs[1];
  float *dst = p.outputs[0];
  auto num_indices = module::getNumElements(getIndices());
  auto ax = getAxis();
  int64_t outer_dims = 1;
  int64_t inner_dims = 1;
  auto input_shape = module::getShape(getInput());
  for (int i = 0; i < ax; ++i) {
    outer_dims *= input_shape[i];
  }
  for (int i = ax + 1; i < input_shape.size(); ++i) {
    inner_dims *= input_shape[i];
  }

  auto num_elems = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elems))
  for (int64_t i = 0; i < outer_dims; ++i) {
    for (int64_t j = 0; j < num_indices; ++j) {
      for (int64_t k = 0; k < inner_dims; ++k) {
        int64_t src_idx = (i * input_shape[ax] + (int64_t)inds[j]) * inner_dims + k;
        int64_t dst_idx = (i * num_indices + j) * inner_dims + k;
        dst[dst_idx] = src[src_idx];
      }
    }
  }

  return success();
}

mlir::Type tpu::GatherOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // indices
    auto opd = op->getOperand(1);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    auto stype = module::getStorageType(opd);
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwidth = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwidth);
  }
  return type_verify_case_same(op, opd_idx, mode);
}
