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

void RoiExtractorLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::RoiExtractorOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RoiExtractorLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::RoiExtractorOp op,
                                        bool asymmetric) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
