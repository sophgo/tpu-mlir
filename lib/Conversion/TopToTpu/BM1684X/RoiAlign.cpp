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

void RoiAlignLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::RoiAlignOp op) const {
  lowering_common_f32<tpu::RoiAlignOp>(rewriter, op);
}

void RoiAlignLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::RoiAlignOp op) const {
  LoweringF32(rewriter, op);
}

void RoiAlignLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::RoiAlignOp op) const {
  LoweringF32(rewriter, op);
}

void RoiAlignLowering::LoweringF8(PatternRewriter &rewriter,
                                  top::RoiAlignOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RoiAlignLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::RoiAlignOp op, bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void RoiAlignLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::RoiAlignOp op, bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void RoiAlignLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::RoiAlignOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
