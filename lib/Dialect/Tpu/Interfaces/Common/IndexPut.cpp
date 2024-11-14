//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::IndexPutOp::init(InferenceParameter &p) { return success(); }
void tpu::IndexPutOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::IndexPutOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  const float *inds = p.inputs[1];
  const float *vals = p.inputs[2];
  float *dst = p.outputs[0];
  auto input_shape = module::getShape(getInput());
  int64_t input_num = module::getNumElements(getInput());
  int64_t num_indices = module::getNumElements(getIndices());

  int64_t inner_dims = std::accumulate(
      input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int>());
  std::memcpy(dst, src, input_num * sizeof(float));
#pragma omp parallel for schedule(static, omp_schedule(num_indices))
  for (int64_t i = 0; i < num_indices; ++i) {
    for (int64_t j = 0; j < inner_dims; ++j) {
      int64_t dst_idx = (inds[i] * inner_dims) + j;
      int64_t val_idx = (i * inner_dims) + j;
      if (getAccumulate()) {
        dst[dst_idx] += vals[val_idx];
      } else
        dst[dst_idx] = vals[val_idx];
    }
  }
  return success();
}

mlir::Type tpu::IndexPutOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
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
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      // indices should be int32 in BM1684x
      bitwidth = 32;
    }
    return Builder(op).getIntegerType(bitwidth);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::IndexPutOp::support_multi_core() { return false; }
