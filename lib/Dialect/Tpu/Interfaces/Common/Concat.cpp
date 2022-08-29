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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::ConcatOp::init(InferenceParameter &p) { return success(); }
void tpu::ConcatOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ConcatOp::inference(InferenceParameter &p) {
  auto axis_ = axis();
  auto op0_shape = inputs()[0].getType().cast<RankedTensorType>().getShape();

  int64_t high = 1;
  for (int64_t i = 0; i < axis_; ++i)
    high *= op0_shape[i];

  SmallVector<int64_t> tailNum(inputs().size());
  for (auto idt : llvm::enumerate(inputs())) {
    tailNum[idt.index()] =
        idt.value().getType().cast<RankedTensorType>().getNumElements() / high;
  }
  auto out_p = p.outputs[0];
  for (int64_t i = 0; i < high; ++i) {
    for (auto idt : llvm::enumerate(tailNum)) {
      memcpy(out_p, p.inputs[idt.index()] + i * idt.value(),
             idt.value() * sizeof(float));
      out_p += idt.value();
    }
  }
  return success();
}

LogicalResult tpu::ConcatOp::LocalGenSupport() {
  auto shape = Module::getShape(output());
  int num_dims = shape.size();
  auto ax = axis();
  if (ax == 1 && (num_dims == 3 || num_dims == 4)) {
    return success();
  }
  return failure();
}
