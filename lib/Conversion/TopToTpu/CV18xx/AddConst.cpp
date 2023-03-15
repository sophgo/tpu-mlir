//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-add-const"

namespace tpu_mlir {
namespace cv18xx {

void AddConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::AddConstOp op, bool asymmetric) const {
  llvm_unreachable("Not supported now");
}

void AddConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::AddConstOp op) const {
  llvm_unreachable("Not supported now");
}

} // namespace cv18xx
} // namespace tpu_mlir
