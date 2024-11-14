//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MaxUnpoolOp::init(InferenceParameter &p) {
  return success();
}
void tpu::MaxUnpoolOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MaxUnpoolOp::inference(InferenceParameter &p) {
  int64_t N, C, H, W;
  module::getNCHW(getInput(), N, C, H, W);
  auto scale_h_ = getScaleH();
  auto scale_w_ = getScaleW();
  int64_t OH = H * scale_h_;
  int64_t OW = W * scale_w_;
  auto num_elem = module::getNumElements(getOutput());

  int64_t NC = N * C;
  std::fill_n(p.outputs[0], num_elem, 0.0f);
#pragma omp parallel for schedule(static, omp_schedule(NC))
  for (int idx = 0; idx < NC; ++idx) {
    auto input_data = p.inputs[0] + idx * H * W;
    auto mask_data = p.inputs[1] + idx * H * W;
    auto output_data = p.outputs[0] + idx * OH * OW;
    for (int i = 0; i < H * W; ++i) {
      int offset = static_cast<int>(mask_data[i]);
      if (offset >= H * W * scale_h_ * scale_w_) {
        llvm_unreachable("out of range");
      }
      output_data[offset] = input_data[i];
    }
  }
  return success();
}

LogicalResult tpu::MaxUnpoolOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto unit = getScaleH();
  if (out_idx % unit || out_slice % unit) {
    return failure();
  }
  in_idx = out_idx / unit;
  in_slice = out_slice / unit;
  return success();
}

LogicalResult tpu::MaxUnpoolOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto unit = getScaleW();
  if (out_idx % unit || out_slice % unit) {
    return failure();
  }
  in_idx = out_idx / unit;
  in_slice = out_slice / unit;
  return success();
}

void tpu::MaxUnpoolOp::assign_sec_info(int64_t n_step, int64_t c_step,
                                       int64_t h_step, int64_t d_step,
                                       int64_t w_step, group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

LogicalResult tpu::MaxUnpoolOp::LocalGenSupport() { return failure(); }

bool tpu::MaxUnpoolOp::support_multi_core() { return false; }
