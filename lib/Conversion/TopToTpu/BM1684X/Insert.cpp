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

void InsertLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::InsertOp op) const {
  lowering_common_f32<tpu::InsertOp>(rewriter, op);
}

void InsertLowering::LoweringINT8(PatternRewriter &rewriter, top::InsertOp op,
                                  bool asymmetric) const {
  lowering_common_int8<tpu::InsertOp>(rewriter, op.getOperation(), asymmetric);
}
void InsertLowering::LoweringINT4(PatternRewriter &rewriter, top::InsertOp op,
                                  bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void InsertLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::InsertOp op) const {
  lowering_common_bf16<tpu::InsertOp>(rewriter, op);
}

void InsertLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::InsertOp op) const {
  lowering_common_f16<tpu::InsertOp>(rewriter, op);
}

void InsertLowering::LoweringF8(PatternRewriter &rewriter,
                                top::InsertOp op) const {
  bool isE4 = module::getMode() == module::Mode::F8E4M3;
  lowering_common_f8<tpu::InsertOp>(rewriter, op, isE4);
}

void InsertLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::InsertOp op) const {
  lowering_common<tpu::InsertOp>(rewriter, op.getOperation(),
                                 op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
