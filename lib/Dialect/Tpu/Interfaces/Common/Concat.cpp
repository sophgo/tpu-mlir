//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
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
  if (module::isCV18xx()) {
    if (ax == 1 && (num_dims == 3 || num_dims == 4)) {
      return success();
    }
  } else if (module::isBM1684Family()) {
    auto status = success();
    auto runMode = getRunMode(getOperation());
    if (ax > 3 ||
        (!module::getStorageType(getOutput()).isInteger(32) && ax == 0)) {
      status = failure();
    }
    if (runMode == RunMode::TPU_DYNAMIC && (ax == 3 && num_dims > 4)) {
      status = failure();
    }
    return status;
  } else {
    return ax > 3 || ax == 0 ? failure() : success();
  }
  return failure();
}

void tpu::ConcatOp::assign_fw_param(void *param) {
  fw_concat_layer_param_t concat_param = {0};
  concat_param.concat_axis = getAxis();
  concat_param.input_num = getInputs().size();
  concat_param.base_dims = module::getShape(getInputs()[0]).size();
  for (int i = 0; i < concat_param.input_num; ++i) {
    module::getGlobalShape(getInputs()[i], concat_param.base_shape);
  }
  memcpy(param, &concat_param, sizeof(fw_concat_layer_param_t));
}

LogicalResult tpu::ConcatOp::AllowDataSplit(int64_t axis,
                                            group_type_t group_type) {
  return getAxis() == axis ? failure() : success();
}
