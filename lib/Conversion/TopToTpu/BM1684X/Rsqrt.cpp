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

static void set_rsqrt_attr(PatternRewriter &rewriter, top::RsqrtOp op) {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::RSQRT));
}

void RsqrtLowering::LoweringF32(PatternRewriter &rewriter,
                                top::RsqrtOp op) const {
  set_rsqrt_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void RsqrtLowering::LoweringINT8(PatternRewriter &rewriter, top::RsqrtOp op,
                                 bool asymmetric) const {
  Value table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                          [](double val) { return 1.f / std::sqrt(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void RsqrtLowering::LoweringINT4(PatternRewriter &rewriter, top::RsqrtOp op,
                                 bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void RsqrtLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::RsqrtOp op) const {
  set_rsqrt_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void RsqrtLowering::LoweringF16(PatternRewriter &rewriter,
                                top::RsqrtOp op) const {
  set_rsqrt_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  else
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void RsqrtLowering::LoweringF8(PatternRewriter &rewriter,
                               top::RsqrtOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void RsqrtLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::RsqrtOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, false);
}

} // namespace bm1684x
} // namespace tpu_mlir
