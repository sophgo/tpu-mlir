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
  auto chip = Module::getChip(getOperation());
  bool is_cv18xx = Module::isCV18xx(chip);
  auto nInputs = inputs().size();
  // allocate tmp input
  std::vector<float *> tmp_inputs(nInputs);
  for (int i = 0; i < nInputs; ++i) {
    auto num_elem = Module::getNumElements(inputs()[i]);
    tmp_inputs[i] = new float[num_elem];
    memcpy(tmp_inputs[i], p.inputs[i], num_elem * sizeof(float));
  }

  if (is_cv18xx && Quant::isUniformQuantized(output())) {
    auto out_type = Module::getStorageType(output());
    auto multiplier_v = Module::getI64Array(multipliers(), nInputs, 1);
    auto rshift_v = Module::getI64Array(rshifts(), nInputs, 0);
    for (int idx = 0; idx < nInputs; ++idx) {
      if (multiplier_v->at(idx) == 1 && rshift_v->at(idx) == 0) {
        continue;
      }
      auto num_elem = Module::getNumElements(inputs()[idx]);
      auto &inp = tmp_inputs[idx];
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; ++i) {
        inp[i] = applyMultiplierAndRShift(inp[i], multiplier_v->at(idx),
                                          rshift_v->at(idx), CVI_QUANT);
        inp[i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(inp[i])
                                               : Quant::to_int8(inp[i]);
      }
    }
  }
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
      memcpy(out_p, tmp_inputs[idt.index()] + i * idt.value(),
             idt.value() * sizeof(float));
      out_p += idt.value();
    }
  }

  if (do_relu()) {
    auto limit = relu_limit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], Module::getNumElements(output()),
                  limit);
  }

  // free tmp input
  for (int i = 0; i < nInputs; ++i) {
    delete[] tmp_inputs[i];
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
