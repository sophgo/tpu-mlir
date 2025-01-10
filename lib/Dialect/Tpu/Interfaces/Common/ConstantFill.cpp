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

LogicalResult tpu::ConstantFillOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ConstantFillOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ConstantFillOp::inference(InferenceParameter &p) {
  if (module::isWeight(getInput())) {
    auto weight = cast<top::WeightOp>(getInput().getDefiningOp());
    const auto shape = weight.read<float>();
    std::vector<int64_t> shape_(shape->begin(), shape->end());
    int idx = 0;
    for (auto a : shape_) {
      if (a == -1)
        shape_[idx] = (int64_t)1;
      idx += 1;
    }
    module::setShape(getOutput(), shape_);
  } else {
    auto num_elem = module::getNumElements(getInput());
    std::vector<int64_t> out_shape;
    for (int i = 0; i < num_elem; i++) {
      out_shape.push_back(p.inputs[0][i]);
    }
    module::setShape(getOutput(), out_shape);
  }
  auto num_elem = module::getNumElements(getOutput());
  float const_val = getValue().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int i = 0; i < num_elem; ++i) {
    p.outputs[0][i] = const_val;
  }
  return success();
}

void tpu::ConstantFillOp::assign_fw_param(void *param) {
  fw_constant_fill_layer_param_t fw_constant_fill_layer_param = {0};
  fw_constant_fill_layer_param.value = getValue().convertToDouble();
  fw_constant_fill_layer_param.dtype = DTYPE_FP32;
  fw_constant_fill_layer_param.type_len = 4;
  memcpy(param, &fw_constant_fill_layer_param,
         sizeof(fw_constant_fill_layer_param));
}

mlir::Type tpu::ConstantFillOp::type_verify(uint64_t opd_idx,
                                            TypeCastMode &mode) {
  return do_nothing(mode);
}

bool tpu::ConstantFillOp::support_multi_core() { return false; }
