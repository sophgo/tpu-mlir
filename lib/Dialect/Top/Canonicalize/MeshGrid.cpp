
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

using namespace tpu_mlir::top;

// MeshGrid canonicalization disabled: MeshGrid2Mul was broken (generated
// invalid MulOp with single operand). MeshGrid has a full CUDA backend path.
void MeshGridOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
}
