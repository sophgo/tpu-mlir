//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void ClipLowering::LoweringF32(PatternRewriter &rewriter, top::ClipOp op) const {
  lowering_common_f32<tpu::ClipOp>(rewriter, op);
}
void ClipLowering::LoweringINT4(PatternRewriter &rewriter, top::ClipOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ClipLowering::LoweringINT8(PatternRewriter &rewriter, top::ClipOp op,
                               bool asymmetric) const {
  // nodechip fix8b to be implemented,
  LoweringF16(rewriter, op);
}

void ClipLowering::LoweringBF16(PatternRewriter &rewriter, top::ClipOp op) const {
  lowering_common_bf16<tpu::ClipOp>(rewriter, op);
}

void ClipLowering::LoweringF16(PatternRewriter &rewriter, top::ClipOp op) const {
  lowering_common_f16<tpu::ClipOp>(rewriter, op);
}

void ClipLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ClipOp op) const {
  LoweringF16(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
