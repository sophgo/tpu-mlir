//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

using namespace mlir;
using namespace tpu_mlir::top;

void MaxPoolOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  // do nothing
}
