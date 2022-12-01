//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::UnpackOp::getFLOPs() { return 0; }

LogicalResult top::UnpackOp::init(InferenceParameter &p) { return success(); }
void top::UnpackOp::deinit(InferenceParameter &p) {}

LogicalResult top::UnpackOp::inference(InferenceParameter &p) {
  auto axis_ = axis();
  auto num_ = num();
  auto in_shape = Module::getShape(input());

  int64_t high = 1, inner = 1;
  for (int64_t i = 0; i < axis_; ++i)
    high *= in_shape[i];
  for (int64_t i = axis_; i < in_shape.size(); ++i) {
    inner *= in_shape[i];
  }

  // auto out_p = p.outputs[0];
  // for (int64_t i = 0; i < high; ++i) {
  //   for (int64_t j = 0; j < num_; ++j) {
  //     memcpy(out_p, p.inputs[idt.index()] + i * idt.value(),
  //            idt.value() * sizeof(float));
  //     out_p += idt.value();
  //   }
  // }
#pragma omp parallel for schedule(static, omp_schedule(high))
  for (int i = 0; i < high; ++i) {
    int64_t index = i * in_shape[axis_] * inner;
    for (int j = 0; j < num_; ++j) {
      memcpy(p.outputs[j] + i * inner,
             p.inputs[0] + index + j * inner,
             inner * sizeof(float));
    }
  }

  return success();
}
