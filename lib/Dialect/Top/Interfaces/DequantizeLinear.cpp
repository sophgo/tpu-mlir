//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::DequantizeLinearOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::DequantizeLinearOp::init(InferenceParameter &p) {
  return success();
}
void top::DequantizeLinearOp::deinit(InferenceParameter &p) {}

LogicalResult top::DequantizeLinearOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  auto shape = getInput().getType().cast<RankedTensorType>().getShape();
  auto zero_point = module::getI32Array(getXZeroPoint());
  auto raw_zero_point = *zero_point;
  auto scale = module::getF64Array(getXScale());
  auto raw_scale = *scale;
  ASSERT_THIS(raw_scale.size() == raw_zero_point.size() &&
              "zero point & scale size missmatch");
  if (raw_zero_point.size() == 1) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      auto val = p.inputs[0][i];
      p.outputs[0][i] = (val - raw_zero_point[0]) * raw_scale[0];
    }
  } else {
    ASSERT_THIS(getAxis() == 0 && "Cannot handle axis!=0");
    ASSERT_THIS(raw_scale.size() == shape[getAxis()] &&
                "zero point & input shape missmatch");
    int64_t res = 1;
    for (int i = 1; i < shape.size(); i++)
      res *= shape[i];
#pragma omp parallel for schedule(static, omp_schedule(shape[0]))
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0; j < res; j++) {
        ASSERT_THIS(raw_zero_point[i] == 0 &&
                    "Cannot support per channel zero point dequant.");
        auto val = p.inputs[0][i * res + j];
        p.outputs[0][i * res + j] = (val - raw_zero_point[i]) * raw_scale[i];
      }
    }
  }
  return success();
}

void top::DequantizeLinearOp::shape_inference() {
  common_shape_inference(getOperation());
}
