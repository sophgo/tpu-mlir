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

void C2CAllReduceLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::C2CAllReduceOp op) const {
  lowering_common_f32<tpu::C2CAllReduceOp>(rewriter, op.getOperation(), 2);
}

void C2CAllReduceLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::C2CAllReduceOp op,
                                        bool asymmetric) const {
  lowering_common_int8<tpu::C2CAllReduceOp>(rewriter, op.getOperation(),
                                            asymmetric, 2);
}

void C2CAllReduceLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::C2CAllReduceOp op,
                                        bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void C2CAllReduceLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::C2CAllReduceOp op) const {
  lowering_common_bf16<tpu::C2CAllReduceOp>(rewriter, op.getOperation(), 2);
}

void C2CAllReduceLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::C2CAllReduceOp op) const {
  lowering_common_f16<tpu::C2CAllReduceOp>(rewriter, op.getOperation(), 2);
}

void C2CAllReduceLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::C2CAllReduceOp op) const {
  llvm_unreachable("C2CAllReduce does not support F8");
}

void C2CAllReduceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::C2CAllReduceOp op) const {
  LoweringF16(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
