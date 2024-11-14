//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MaskRCNNMaskPoolerOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void tpu::MaskRCNNMaskPoolerOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::MaskRCNNMaskPoolerOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

mlir::Type tpu::MaskRCNNMaskPoolerOp::type_verify(uint64_t opd_idx,
                                                  TypeCastMode &mode) {
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

bool tpu::MaskRCNNMaskPoolerOp::support_multi_core() { return false; }
