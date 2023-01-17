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

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ConcatOp::init(InferenceParameter &p) { return success(); }
void tpu::ConcatOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ConcatOp::inference(InferenceParameter &p) {
  auto axis_ = getAxis();
  bool is_cv18xx = module::isCV18xx();
  auto nInputs = getInputs().size();
  // allocate tmp input
  std::vector<float *> tmp_inputs(nInputs);
  for (int i = 0; i < nInputs; ++i) {
    auto num_elem = module::getNumElements(getInputs()[i]);
    tmp_inputs[i] = new float[num_elem];
    memcpy(tmp_inputs[i], p.inputs[i], num_elem * sizeof(float));
  }

  if (is_cv18xx && module::isUniformQuantized(getOutput())) {
    auto out_type = module::getStorageType(getOutput());
    auto multiplier_v = module::getI64Array(getMultipliers(), nInputs, 1);
    auto rshift_v = module::getI64Array(getRshifts(), nInputs, 0);
    for (int idx = 0; idx < nInputs; ++idx) {
      if (multiplier_v->at(idx) == 1 && rshift_v->at(idx) == 0) {
        continue;
      }
      auto num_elem = module::getNumElements(getInputs()[idx]);
      auto &inp = tmp_inputs[idx];
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; ++i) {
        inp[i] = applyMultiplierAndRShift(inp[i], multiplier_v->at(idx),
                                          rshift_v->at(idx));
        inp[i] = saturate(inp[i], out_type);
      }
    }
  }
  auto op0_shape = getInputs()[0].getType().cast<RankedTensorType>().getShape();

  int64_t high = 1;
  for (int64_t i = 0; i < axis_; ++i)
    high *= op0_shape[i];

  SmallVector<int64_t> tailNum(getInputs().size());
  for (auto idt : llvm::enumerate(getInputs())) {
    tailNum[idt.index()] =
        idt.value().getType().cast<RankedTensorType>().getNumElements() / high;
  }
  auto out_p = p.outputs[0];
  for (int64_t i = 0; i < high; ++i) {
    for (auto idt : llvm::enumerate(tailNum)) {
      memcpy(out_p, tmp_inputs[idt.index()] + i * idt.value(),
             idt.value() * sizeof(float));
      out_p += idt.value();
    }
  }

  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0],
                  module::getNumElements(getOutput()), limit);
  }

  // free tmp input
  for (int i = 0; i < nInputs; ++i) {
    delete[] tmp_inputs[i];
  }
  return success();
}

LogicalResult tpu::ConcatOp::LocalGenSupport() {
  auto shape = module::getShape(getOutput());
  int num_dims = shape.size();
  auto ax = getAxis();
  if (ax == 1 && (num_dims == 3 || num_dims == 4)) {
    return success();
  }
  return failure();
}
