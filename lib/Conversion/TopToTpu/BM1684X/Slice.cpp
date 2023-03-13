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

void SliceLowering::LoweringF32(PatternRewriter &rewriter,
                                top::SliceOp op) const {
  lowering_common_f32<tpu::SliceOp>(rewriter, op, 2);
}
void SliceLowering::LoweringINT4(PatternRewriter &rewriter, top::SliceOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SliceLowering::LoweringINT8(PatternRewriter &rewriter, top::SliceOp op,
                                 bool asymmetric) const {
  lowering_common_int8<tpu::SliceOp>(rewriter, op, asymmetric, 2);
}

void SliceLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::SliceOp op) const {
  auto out = op.getOutput();
  if (module::isCalibratedType(out)) {
    //For fuse_preprocess(crop image) use, it should be lowered to uint8.
    auto qtype = module::getCalibratedType(out);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    if (min == 0 && max == 255) {
      lowering_common_int8<tpu::SliceOp>(rewriter, op, false);
      return;
    }
  }
  lowering_common_bf16<tpu::SliceOp>(rewriter, op, 2);
}

void SliceLowering::LoweringF16(PatternRewriter &rewriter,
                                top::SliceOp op) const {
  lowering_common_f16<tpu::SliceOp>(rewriter, op, 2);
}

void SliceLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::SliceOp op) const {
  lowering_common<tpu::SliceOp>(rewriter, op, op.getOutput().getType(), 2);
}

} // namespace bm1684x
} // namespace tpu_mlir
