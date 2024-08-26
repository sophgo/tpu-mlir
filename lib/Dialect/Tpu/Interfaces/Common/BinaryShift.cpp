//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"

LogicalResult tpu::BinaryShiftOp::init(InferenceParameter &p) {
  int index0 = 0, index1 = 1;
  if (getIsReverse() && getModeAttr().str() == "Sub") {
    index0 = 1, index1 = 0;
  }
  auto lhs_shape = module::getShape(*(getODSOperands(index0).begin()));
  auto rhs_shape = module::getShape(*(getODSOperands(index1).begin()));

  std::map<std::string, algorithm> map_mode = {
      {"Add", algorithm::binary_add},
      {"Sub", algorithm::binary_sub},
      {"Mul", algorithm::binary_mul}};

  auto iter = map_mode.find(getModeAttr().str());
  algorithm binary_mode;
  if (iter != map_mode.end()) {
    binary_mode = iter->second;
  }

  auto binary = new Binary();
  (*binary)
      .hs(p.inputs[index0], p.inputs[index1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(binary_mode)
      .setup();
  p.handle = (void *)binary;
  return success();
}

void tpu::BinaryShiftOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::BinaryShiftOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());

  auto binary = (Binary *)p.handle;
  binary->run();
  int32_t shift_val = getShift();
  auto rmode = round_mode_convert(getRoundMode());
  // bool is_satu = getSaturation();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int i = 0; i < num_elem; i++) {
    int64_t sum = p.outputs[0][i];
    sum = RightShiftRound(sum, -shift_val, rmode);
    p.outputs[0][i] = saturate(sum, out_type);
  }
  return success();
}

LogicalResult tpu::BinaryShiftOp::LocalGenSupport() {
  return BroadCastBinaryLocalGenSupport(getOperation());
}

ArrayAttr tpu::BinaryShiftOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::BinaryShiftOp::support_multi_core() { return false; }
