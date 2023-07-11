//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-add"

namespace tpu_mlir {
namespace cv18xx {
void RMSNormLowering::LoweringINT8(PatternRewriter &rewriter, top::RMSNormOp op,
                                   bool asymmetric) const {
  llvm_unreachable("to be implemented");
}

void RMSNormLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::RMSNormOp op) const {
  llvm_unreachable("to be implemented");
}

} // namespace cv18xx
} // namespace tpu_mlir
