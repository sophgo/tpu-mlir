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

void SqueezeLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SqueezeOp op) const {
  lowering_common_f32<tpu::SqueezeOp>(rewriter, op);
}

void SqueezeLowering::LoweringINT8(PatternRewriter &rewriter, top::SqueezeOp op,
                                   bool asymmetric) const {
  lowering_common_int8<tpu::SqueezeOp>(rewriter, op, asymmetric);
}
void SqueezeLowering::LoweringINT4(PatternRewriter &rewriter, top::SqueezeOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SqueezeLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SqueezeOp op) const {
  lowering_common_bf16<tpu::SqueezeOp>(rewriter, op);
}

void SqueezeLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SqueezeOp op) const {
  lowering_common_f16<tpu::SqueezeOp>(rewriter, op);
}

void SqueezeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::SqueezeOp op) const {
  lowering_common<tpu::SqueezeOp>(rewriter, op, op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
