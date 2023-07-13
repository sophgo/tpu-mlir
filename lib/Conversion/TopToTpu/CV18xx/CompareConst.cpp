//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {

bool is_valid(top::CompareConstOp &op) {
  if (op.getMode().str() != "Equal" ||
      op.getConstVal().convertToDouble() >= 1.e-5) {
    return false;
  }
  return true;
}

void CompareConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::CompareConstOp CompareConstOp,
                                        bool asymmetric) const {
  LoweringBF16(rewriter, CompareConstOp);
}

void CompareConstLowering::LoweringBF16(
    PatternRewriter &rewriter, top::CompareConstOp CompareConstOp) const {
  if (!is_valid(CompareConstOp)) {
    llvm_unreachable("Not support now.");
  }
  lowering_common_bf16<tpu::CompareConstOp>(rewriter, CompareConstOp);
}
} // namespace cv18xx
} // namespace tpu_mlir
