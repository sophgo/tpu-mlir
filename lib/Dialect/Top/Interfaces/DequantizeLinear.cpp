//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::DequantizeLinearOp::getFLOPs() {
  return Module::getNumElements(output());
}

LogicalResult top::DequantizeLinearOp::init(InferenceParameter &p) {
  return success();
}
void top::DequantizeLinearOp::deinit(InferenceParameter &p) {}

LogicalResult top::DequantizeLinearOp::inference(InferenceParameter &p) {
  auto num_element = Module::getNumElements(output());
  auto shape = input().getType().cast<RankedTensorType>().getShape();
  auto zero_point = Module::getI32Array(x_zero_point());
  auto raw_zero_point = *zero_point;
  auto scale = Module::getF64Array(x_scale());
  auto raw_scale = *scale;
  assert(raw_scale.size() == raw_zero_point.size() &&
         "zero point & scale size missmatch");
  if (raw_zero_point.size() == 1) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      auto val = p.inputs[0][i];
      p.outputs[0][i] = (val - raw_zero_point[0]) * raw_scale[0];
    }
  } else {
    assert(axis() == 0 && "Cannot handle axis!=0");
    assert(raw_scale.size() == shape[axis()] &&
           "zero point & input shape missmatch");
    int64_t res = 1;
    for (int i = 1; i < shape.size(); i++)
      res *= shape[i];
#pragma omp parallel for schedule(static, omp_schedule(shape[0]))
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < res; j++) {
        auto val = p.inputs[0][i * res + j];
        p.outputs[0][i * res + j] = (val - raw_zero_point[i]) * raw_scale[i];
      }
    }
  }
  return success();
}
