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

void C2CBroadcastLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::C2CBroadcastOp op) const {
  lowering_common_f32<tpu::C2CBroadcastOp>(rewriter, op.getOperation(), 1);
}

void C2CBroadcastLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::C2CBroadcastOp op,
                                        bool asymmetric) const {
  lowering_common_int8<tpu::C2CBroadcastOp>(rewriter, op.getOperation(),
                                            asymmetric, 1);
}

void C2CBroadcastLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::C2CBroadcastOp op,
                                        bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void C2CBroadcastLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::C2CBroadcastOp op) const {
  lowering_common_bf16<tpu::C2CBroadcastOp>(rewriter, op.getOperation(), 1);
}

void C2CBroadcastLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::C2CBroadcastOp op) const {
  lowering_common_f16<tpu::C2CBroadcastOp>(rewriter, op.getOperation(), 1);
}

void C2CBroadcastLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::C2CBroadcastOp op) const {
  llvm_unreachable("C2CBroadcast does not support F8");
}

void C2CBroadcastLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::C2CBroadcastOp op) const {
  LoweringF16(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
