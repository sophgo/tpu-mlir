//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::BinaryConstShiftOp::init(InferenceParameter &p) {
  return success();
}

void tpu::BinaryConstShiftOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::BinaryConstShiftOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  int64_t const_val = getScale();
  int32_t shift_val = getShift();
  auto rmode = round_mode_convert(getRoundMode());
  // bool is_satu = getSaturation();
  if (getMode().str() == "Add") {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      int64_t sum = p.inputs[0][i] + const_val;
      sum = RightShiftRound(sum, -shift_val, rmode);
      p.outputs[0][i] = saturate(sum, out_type, rmode);
    }
  } else if (getMode().str() == "Sub") {
    if (getIsReverse() == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        int64_t sum = p.inputs[0][i] - const_val;
        sum = RightShiftRound(sum, -shift_val, rmode);
        p.outputs[0][i] = saturate(sum, out_type, rmode);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        int64_t sum = const_val - p.inputs[0][i];
        sum = RightShiftRound(sum, -shift_val, rmode);
        p.outputs[0][i] = saturate(sum, out_type, rmode);
      }
    }
  } else if (getMode().str() == "Mul") {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      int64_t sum = p.inputs[0][i] * const_val;
      sum = RightShiftRound(sum, -shift_val, rmode);
      p.outputs[0][i] = saturate(sum, out_type, rmode);
    }
  } else {
    UNREACHABLE_THIS("Not Implemented");
  }

  return success();
}

void tpu::BinaryConstShiftOp ::assign_sec_info(int64_t n_step, int64_t c_step,
                                               int64_t h_step, int64_t d_step,
                                               int64_t w_step,
                                               group_type_t group_type,
                                               local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  int64_t n, c, d, h, w, on, oc, od, oh, ow;
  auto input = getOperand();
  auto output = getResult();
  module::getNCDHW(input, n, c, d, h, w, group_type);
  module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input, n_step, h_step, d_step,
                                               w_step, c_step);
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = in_gi.d_slice;
  sec_info.h_slice = gi.h_slice;
  sec_info.w_slice = gi.w_slice;
  sec_info.c_slice = gi.c_slice;
  sec_info.n_idx = in_gi.n_idx;
  sec_info.d_idx = in_gi.d_idx;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info.w_idx = in_gi.w_idx;
  sec_info.is_w_split = !(in_gi.w_idx == 0 && in_gi.w_slice == w);
  sec_info.c_idx = gi.c_idx;
  sec_info.is_c_split = !(in_gi.c_idx == 0 && in_gi.c_slice == c);
  // set margins
  setHWMargins(sec_info.hw_margins_opdA, in_gi, gi);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
}

ArrayAttr tpu::BinaryConstShiftOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::BinaryConstShiftOp::support_multi_core() { return false; }
