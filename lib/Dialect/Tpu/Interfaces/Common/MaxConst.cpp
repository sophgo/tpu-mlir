//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MaxConstOp::init(InferenceParameter &p) { return success(); }

void tpu::MaxConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MaxConstOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
  auto const_val = (float)getConstVal().convertToDouble();
  if (module::isUniformQuantized(getOutput())) {
    auto i_qtype = module::getUniformQuantizedType(getInput());
    double izp = 0;
    if (module::isUniformQuantized(getInput()))
      izp = i_qtype.getZeroPoint();
    auto out_qtype = module::getUniformQuantizedType(getOutput());
    double ozp = 0;
    auto out_type = module::getStorageType(getOutput());
    if (asym)
      ozp = out_qtype.getZeroPoint();
    // #pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      float p_val = p.inputs[0][i] - izp;
      p_val = applyMultiplierAndRShift(p_val, getMultiplier(), 0);
      p_val = std::max(p_val, const_val);
      p_val = applyMultiplierAndRShift(p_val, 1, getRshift());
      if (getDoRelu() && p_val < 0) {
        p_val = 0;
      }
      p.outputs[0][i] = saturate(p_val, out_type);
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = std::max(p.inputs[0][i], const_val);
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

LogicalResult tpu::MaxConstOp::LocalGenSupport() { return success(); }

void tpu::MaxConstOp::assign_fw_param(void *param) {
  IR_PARAM_CONST_BINARY(BINARY_MAX);
}

ArrayAttr tpu::MaxConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::MaxConstOp::support_multi_core() { return false; }
