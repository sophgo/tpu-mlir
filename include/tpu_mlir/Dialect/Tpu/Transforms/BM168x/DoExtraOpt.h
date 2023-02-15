//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/PatternMatch.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;

namespace tpu_mlir {

namespace bm1684x {
   // add bm1684x unique patterns here

void populateDoExtraOptPatterns(RewritePatternSet *patterns);
} // namespace bm1684x
} // namespace tpu_mlir
