//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/DoExtraOpt.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;
namespace tpu_mlir {

namespace cv18xx {

void populateDoExtraOptPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      FuseLeakReluPattern,
      MoveConvStrideToEltwiseOpPattern,
      SplitReluLimitPattern,
      SplitReducePattern
  >(patterns->getContext());
  // clang-format on
};

} // namespace cv18xx
} // namespace tpu_mlir
