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
void RoiExtractorLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::RoiExtractorOp op,
                                        bool asymmetric) const {
  UNREACHABLE_OP("to be implemented", op);
}

void RoiExtractorLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::RoiExtractorOp op) const {
  UNREACHABLE_OP("to be implemented", op);
}

} // namespace cv18xx
} // namespace tpu_mlir
