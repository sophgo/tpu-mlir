//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void RMSNormLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::RMSNormOp op) const {
  llvm_unreachable("to be implemented");
}

void RMSNormLowering::LoweringINT8(PatternRewriter &rewriter, top::RMSNormOp op,
                                   bool asymmetric) const {
  llvm_unreachable("to be implemented");
}

} // namespace bm1684
} // namespace tpu_mlir
