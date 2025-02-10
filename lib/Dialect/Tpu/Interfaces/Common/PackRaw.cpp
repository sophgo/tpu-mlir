//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::PackRawOp::init(InferenceParameter &p) {
  auto in_type = module::getStorageType(getInput());
  // auto out_type = module::getStorageType(getOutput());
  assert(in_type.isInteger(8));
  return success();
}

void tpu::PackRawOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::PackRawOp::inference(InferenceParameter &p) {
  // float white_level = getWhiteLevel().convertToDouble();
  // float black_level = getBlackLevel().convertToDouble();
  // std::vector<int64_t> in_shape = module::getShape(getInput());
  // std::vector<int64_t> out_shape = module::getShape(getOutput());
  // int n  = in_shape[0];
  // int ic = in_shape[1];
  // int ih = in_shape[2];
  // int iw = in_shape[3];
  // assert ( ic == 1 );
  // assert ( ih % 2 == 0 );
  // assert ( ih % 3 == 0 );
  // int block_size[2] = { 2 , 2 };
  // int bh = block_size[0];
  // int bw = block_size[1];
  // int oh = ih / bh;
  // int ow = iw / 3;
  // int ph = ( 32 - ( oh % 32 ) ) % 32;
  // int pw = ( 32 - ( ow % 32 ) ) % 32;
  // assert ( out_shape[1] == bh * bw );
  // assert ( out_shape[2] == oh + ph );
  // assert ( out_shape[3] == ow + pw );

  return success();
}

mlir::Type tpu::PackRawOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}

bool tpu::PackRawOp::support_multi_core() { return false; }
