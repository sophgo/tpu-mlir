//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

LogicalResult tpu::BinaryShiftOp::init(InferenceParameter &p) {
  int index0 = 0, index1 = 1;
  if (getIsReverse() && getModeAttr().str() == "Sub") {
    index0 = 1, index1 = 0;
  }
  auto lhs_shape = module::getShape(*(getODSOperands(index0).begin()));
  auto rhs_shape = module::getShape(*(getODSOperands(index1).begin()));

  std::map<std::string, algorithm> map_mode = {{"Add", algorithm::binary_add},
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

void tpu::BinaryShiftOp::assign_sec_info(int64_t n_step, int64_t c_step,
                                         int64_t h_step, int64_t d_step,
                                         int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  int64_t n0, c0, d0, h0, w0, n1, c1, d1, h1, w1, on, oc, od, oh, ow;
  auto input0 = getOperand(0);
  auto input1 = getOperand(1);
  auto output = getResult();
  module::getNCDHW(input0, n0, c0, d0, h0, w0, group_type);
  module::getNCDHW(input1, n1, c1, d1, h1, w1, group_type);
  module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in0_gi = LocalGenInterface::getGroupInfo(input0, n_step, h_step, d_step,
                                                w_step, c_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(input1, n_step, h_step, d_step,
                                                w_step, c_step);
  sec_info.n_slice = std::max(in0_gi.n_slice, in1_gi.n_slice);
  sec_info.c_slice = std::max(in0_gi.c_slice, in1_gi.c_slice);
  sec_info.d_slice = std::max(in0_gi.d_slice, in1_gi.d_slice);
  sec_info.h_slice = gi.h_slice;
  sec_info.w_slice = gi.w_slice;
  sec_info.n_idx = std::max(in0_gi.n_idx, in1_gi.n_idx);
  sec_info.d_idx = std::max(in0_gi.d_idx, in1_gi.d_idx);
  sec_info.c_idx = std::max(in0_gi.c_idx, in1_gi.c_idx);
  sec_info.is_c_split =
      !(std::max(in0_gi.c_idx, in1_gi.c_idx) == 0 &&
        std::max(in0_gi.c_slice, in1_gi.c_slice) == std::max(c0, c1));
  sec_info.h_idx = std::max(in0_gi.h_idx, in1_gi.h_idx);
  sec_info.is_h_split =
      !(std::max(in0_gi.h_idx, in1_gi.h_idx) == 0 &&
        std::max(in0_gi.h_slice, in1_gi.h_slice) == std::max(h0, h1));
  sec_info.w_idx = std::max(in0_gi.w_idx, in1_gi.w_idx);
  sec_info.is_w_split =
      !(std::max(in0_gi.w_idx, in1_gi.w_idx) == 0 &&
        std::max(in0_gi.w_slice, in1_gi.w_slice) == std::max(w0, w1));
  setHWMargins(sec_info.hw_margins_opdA, in0_gi, gi);
  setHWMargins(sec_info.hw_margins_opdB, in1_gi, gi);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
}

ArrayAttr tpu::BinaryShiftOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::BinaryShiftOp::support_multi_core() { return false; }
