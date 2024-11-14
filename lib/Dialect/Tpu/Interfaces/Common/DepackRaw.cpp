//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::DepackRawOp::init(InferenceParameter &p) {
  return success();
}

void tpu::DepackRawOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::DepackRawOp::inference(InferenceParameter &p) {
  // std::vector<int64_t> in_shape = module::getShape(getInput());
  // std::vector<int64_t> out_shape = module::getShape(getOutput());

  // int block_size[2] = { 2 , 2 };
  // int bh = block_size[0];
  // int bw = block_size[1];

  // int ph = getPaddingH();
  // int pw = getPaddingW();

  // int ic = in_shape[1];
  // int ih = in_shape[2] - ph;
  // int iw = in_shape[3] - pw;
  // assert ( ic == bh * bw );

  // int oh = ih * bh;
  // int ow = iw * bw;

  return success();
}

// no need type verify, DepackRawOp include cast
mlir::Type tpu::DepackRawOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}

bool tpu::DepackRawOp::support_multi_core() { return false; }
