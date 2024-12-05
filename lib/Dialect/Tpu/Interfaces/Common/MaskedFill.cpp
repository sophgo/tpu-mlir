//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MaskedFillOp::init(InferenceParameter &p) {
  return success();
}
void tpu::MaskedFillOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MaskedFillOp::inference(InferenceParameter &p) {
  const auto num_element = module::getNumElements(getOutput());
  const auto in_num_element = module::getNumElements(getOperand(0));
  const auto brn_num_element = module::getNumElements(getOperand(1));
  std::vector<int64_t> output_shape = module::getShape(getOutput());
  std::vector<int64_t> input_shape = module::getShape(getOperand(0));
  std::vector<int64_t> brn_shape = module::getShape(getOperand(1));
  bool in_broadcast = false;
  bool brn_broadcast = false;
  if (in_num_element != num_element) {
    assert(input_shape.size() == output_shape.size());
    in_broadcast = true;
  }
  if (brn_num_element != num_element) {
    assert(brn_shape.size() == output_shape.size());
    brn_broadcast = true;
  }

  const float const_val_ = getConstVal().convertToDouble();
  const float *in = p.inputs[0];
  const float *brn = p.inputs[1];
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    int in_index = i;
    int brn_index = i;
    if (in_broadcast) {
      in_index = getBcastIndex(i, output_shape, input_shape);
    }
    if (brn_broadcast) {
      brn_index = getBcastIndex(i, output_shape, brn_shape);
    }
    const float tbrn = getInversed() ? const_val_ : brn[brn_index];
    const float fbrn = getInversed() ? brn[brn_index] : const_val_;
    p.outputs[0][i] = in[in_index] ? tbrn : fbrn;
  }
  return success();
}

bool tpu::MaskedFillOp::support_multi_core() { return false; }
