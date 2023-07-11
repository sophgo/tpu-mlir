//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

LogicalResult tpu::CompareOp::init(InferenceParameter &p) {

  std::map<std::string, algorithm> map_mode = {
      {"Equal", algorithm::binary_eq},
      {"Greater", algorithm::binary_gt},
      {"GreaterOrEqual", algorithm::binary_ge},
      {"Less", algorithm::binary_lt},
      {"LessOrEqual", algorithm::binary_le},
      {"NotEqual", algorithm::binary_ne},
      {"And", algorithm::binary_mul}};

  auto binary = new Binary();
  auto lhs_shape = module::getShape(getOperand(0));
  auto rhs_shape = module::getShape(getOperand(1));

  auto iter = map_mode.find(getModeAttr().str());
  algorithm compare_mode;
  if (iter != map_mode.end()) {
    compare_mode = iter->second;
  }

  (*binary)
      .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(compare_mode)
      .setup();

  p.handle = (void *)binary;

  return success();
}

void tpu::CompareOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CompareOp::inference(InferenceParameter &p) {

  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();

  return success();
}

LogicalResult tpu::CompareOp::LocalGenSupport() {
  return BroadCastBinaryLocalGenSupport(getOperation());
}
