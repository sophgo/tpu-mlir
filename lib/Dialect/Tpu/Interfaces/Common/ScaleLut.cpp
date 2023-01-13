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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

#include "tpu_mlir/Support/MathUtils.h"


LogicalResult tpu::ScaleLutOp::init(InferenceParameter &p) { return success(); }
void tpu::ScaleLutOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ScaleLutOp::inference(InferenceParameter &p) {
  float *input_data = p.inputs[0];
  float *table = p.inputs[1];
  float *output_data = p.outputs[0];
  std::vector<int64_t> input_shape;
  module::getShapeVec(this->getInput(), input_shape);
  int n = input_shape.at(0);
  int c = input_shape.at(1);
  int h = input_shape.at(2);
  int w = input_shape.at(3);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        int index = ni * c * h * w + ci * h * w + i;
        auto x = input_data[index];
        auto y = table[(int)(ci * 256 + x)];
        output_data[index] = y;
      }
    }
  }
  return success();
}

LogicalResult tpu::ScaleLutOp::LocalGenSupport() {
  if (!module::isCV18xx()) {
    return failure();
  }
  int64_t npu_num = tpu_mlir::backend::CV18xx::NPU_NUM;
  std::vector<int64_t> input_shape;
  module::getShapeVec(this->getInput(), input_shape);
  if (input_shape.size() > 2 && input_shape[1] > npu_num) {
    return failure();
  }
  return success();
}
