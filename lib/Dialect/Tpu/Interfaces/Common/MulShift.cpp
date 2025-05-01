//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::MulShiftOp::init(InferenceParameter &p) { return success(); }
void tpu::MulShiftOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulShiftOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto sType = module::getStorageType(getOutput());
  int64_t in_zp = 0, out_zp = 0;
  if (module::isUniformQuantized(getInput())) {
    auto qtype = module::getUniformQuantizedType(getInput());
    in_zp = qtype.getZeroPoint();
  }
  if (module::isUniformQuantized(getOutput())) {
    auto qtype = module::getUniformQuantizedType(getOutput());
    out_zp = qtype.getZeroPoint();
  }

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    auto v = applyMultiplierAndRShift(p.inputs[0][i] - in_zp,
                                      (int64_t)getMultiplier(), getRshift());
    // should add zp to the outputs.
    v += out_zp;
    p.outputs[0][i] = saturate(v, sType);
  }
  return success();
}

void tpu::MulShiftOp::assign_fw_param(void *param) {
  fw_mulshift_layer_param_t layer_param = {0};
  layer_param.ic = module::getShape(getInput())[1];
  layer_param.mulvalue = getMultiplier();
  layer_param.mulshiftnum = getRshift();
  layer_param.opd0_sign = module::isSign(getInput());
  layer_param.res_sign = module::isSign(getOutput());
  memcpy(param, &layer_param, sizeof(fw_mulshift_layer_param_t));
}

LogicalResult tpu::MulShiftOp::LocalGenSupport() {
  auto input_shape = module::getShape(getInput());
  if (input_shape.size() >= 3 && input_shape[1] > 65535) {
    return failure();
  }
  return success();
}

ArrayAttr tpu::MulShiftOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::MulShiftOp::support_multi_core() { return false; }
