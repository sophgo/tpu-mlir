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

static void set_cos_attr(PatternRewriter &rewriter, top::CosOp op) {
  auto op_ = op.getOperation();
  op_->setAttr("mode",
               tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::COS));
}

void CosLowering::LoweringF32(PatternRewriter &rewriter, top::CosOp op) const {
  set_cos_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void CosLowering::LoweringINT8(PatternRewriter &rewriter, top::CosOp op,
                               bool asymmetric) const {
  Value table = create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                                    [](double val) { return std::cos(val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void CosLowering::LoweringINT4(PatternRewriter &rewriter, top::CosOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void CosLowering::LoweringBF16(PatternRewriter &rewriter, top::CosOp op) const {
  set_cos_attr(rewriter, op);
  if (module::isMARS3() || module::isSGTPUV8()) {
    lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
  } else {
    lowering_common_f32<tpu::ActiveOp>(rewriter, op);
  }
}

void CosLowering::LoweringF16(PatternRewriter &rewriter, top::CosOp op) const {
  set_cos_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

void CosLowering::LoweringF8(PatternRewriter &rewriter, top::CosOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void CosLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::CosOp op) const {
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
