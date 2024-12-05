//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

LogicalResult tpu::LoopOp::init(InferenceParameter &p) { return success(); }

void tpu::LoopOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LoopOp::inference(InferenceParameter &p) {
  return success();
}

Operation::result_range tpu::LoopOp::v_final() {
  auto results = getResults();
  return llvm::make_range(results.begin(),
                          results.begin() + getVInitial().size());
}

Operation::result_range tpu::LoopOp::scan_outputs() {
  auto results = getResults();
  return llvm::make_range(results.begin() + getVInitial().size(),
                          results.end());
}

bool tpu::LoopOp::support_multi_core() { return false; }
