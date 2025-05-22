//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technoroialignies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void RoiExtractorLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::RoiExtractorOp op) const {
  lowering_common_f32<tpu::RoiExtractorOp>(rewriter, op);
}

void RoiExtractorLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::RoiExtractorOp op) const {
  LoweringF32(rewriter, op);
}

void RoiExtractorLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::RoiExtractorOp op) const {
  LoweringF32(rewriter, op);
}

void RoiExtractorLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::RoiExtractorOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RoiExtractorLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::RoiExtractorOp op,
                                        bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void RoiExtractorLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::RoiExtractorOp op,
                                        bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void RoiExtractorLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::RoiExtractorOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
