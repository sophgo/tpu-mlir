//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

#include "tpu_mlir/Support/ActiveUtils.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/LutFunc.h"

static mlir::Type t;

LogicalResult tpu::ActiveOp::init(InferenceParameter &p) { return success(); }
void tpu::ActiveOp::deinit(InferenceParameter &p) {}

static void active_func(InferenceParameter &p, int64_t num, activate_f func) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; ++i) {
    p.outputs[0][i] = func(p.inputs[0][i]);
  }
}

LogicalResult tpu::ActiveOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  t = module::getStorageType(getOutput());
  auto num_element = module::getNumElements(getInput());
  active_func(p, num_element, getActivateFunc(*this));
  if (t.isBF16()) {
    BF16(p.outputs[0], p.outputs[0], num_element);
  } else if (t.isF16()) {
    F16(p.outputs[0], p.outputs[0], num_element);
  }
  return success();
}

LogicalResult tpu::ActiveOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    if (getMode() == ActiveMode::ABSVAL) {
      return success();
    } else {
      return failure();
    }
  }
  return success();
}

void tpu::ActiveOp::assign_fw_param(void *param) {
  fw_active_layer_param_t layer_param = {0};
  layer_param.active_type = (int)getMode();
  layer_param.if_relu = 0; // not implement
  layer_param.relu_upper_limit = 0.f;
  auto shape = module::getShape(getInput());

  layer_param.ic = shape.size() > 1 ? shape[1] : 1;
  layer_param.input_scale_back2float = 1.f;  // not implement
  layer_param.output_scale_back2float = 1.f; // not implement
  layer_param.opd_sign = module::isSign(getInput());
  layer_param.res_sign = module::isSign(getOutput());
  memcpy(param, &layer_param, sizeof(fw_active_layer_param_t));
}

ArrayAttr tpu::ActiveOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::ActiveOp::support_multi_core() { return false; }
