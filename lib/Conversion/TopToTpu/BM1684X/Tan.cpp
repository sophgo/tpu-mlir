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

static void set_tan_attr(PatternRewriter &rewriter, top::TanOp op) {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::TAN));
}

void TanLowering::LoweringF32(PatternRewriter &rewriter, top::TanOp op) const {
  set_tan_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void TanLowering::LoweringINT8(PatternRewriter &rewriter, top::TanOp op,
                               bool asymmetric) const {
  // this op not suitable for int8 quant cuz slight deviation in the former op
  // would result in great difference in tan
  set_tan_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void TanLowering::LoweringINT4(PatternRewriter &rewriter, top::TanOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void TanLowering::LoweringBF16(PatternRewriter &rewriter, top::TanOp op) const {
  set_tan_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void TanLowering::LoweringF16(PatternRewriter &rewriter, top::TanOp op) const {
  set_tan_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void TanLowering::LoweringF8(PatternRewriter &rewriter, top::TanOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void TanLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::TanOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
