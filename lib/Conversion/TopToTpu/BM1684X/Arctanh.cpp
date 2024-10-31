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

static void set_arctanh_attr(PatternRewriter &rewriter, top::ArctanhOp op) {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::ARCTANH));
}

void ArctanhLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::ArctanhOp op) const {
  set_arctanh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void ArctanhLowering::LoweringINT8(PatternRewriter &rewriter, top::ArctanhOp op,
                                   bool asymmetric) const {
  // this op not suitable for int8 quant cuz slight deviation in the former op
  // would result in great difference in arctanh
  set_arctanh_attr(rewriter, op);
  if (module::isMARS3()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}

void ArctanhLowering::LoweringINT4(PatternRewriter &rewriter, top::ArctanhOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void ArctanhLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::ArctanhOp op) const {
  set_arctanh_attr(rewriter, op);
  if (module::isMARS3()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}

void ArctanhLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::ArctanhOp op) const {
  set_arctanh_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void ArctanhLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::ArctanhOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ArctanhLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::ArctanhOp op) const {
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
