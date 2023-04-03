//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DoExtraOpt.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;
namespace tpu_mlir {

namespace bm1684x {

void populateDoExtraOptPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    MatMulHdimBatchPattern,
    MatMulLeftReusePattern,
    PermuteReorderPattern
  >(patterns->getContext());
  // clang-format on
};

} // namespace bm1684x
} // namespace tpu_mlir
