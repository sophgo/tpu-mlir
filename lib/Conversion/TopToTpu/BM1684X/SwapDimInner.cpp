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

void SwapDimInnerLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::SwapDimInnerOp op) const {
  lowering_common_f32<tpu::SwapDimInnerOp>(rewriter, op);
}

void SwapDimInnerLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::SwapDimInnerOp op,
                                        bool asymmetric) const {
  lowering_common_int8<tpu::SwapDimInnerOp>(rewriter, op.getOperation(),
                                            asymmetric);
}
void SwapDimInnerLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::SwapDimInnerOp op,
                                        bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SwapDimInnerLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::SwapDimInnerOp op) const {
  lowering_common_bf16<tpu::SwapDimInnerOp>(rewriter, op);
}

void SwapDimInnerLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::SwapDimInnerOp op) const {
  lowering_common_f16<tpu::SwapDimInnerOp>(rewriter, op);
}

void SwapDimInnerLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::SwapDimInnerOp op) const {
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  lowering_common_f8<tpu::SwapDimInnerOp>(rewriter, op, isE4);
}

void SwapDimInnerLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::SwapDimInnerOp op) const {
  lowering_common<tpu::SwapDimInnerOp>(rewriter, op.getOperation(),
                                       op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
