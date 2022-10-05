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

void PadLowering::LoweringF32(PatternRewriter &rewriter,
                              top::PadOp op) const {
  lowering_common_float<tpu::PadOp>(rewriter, op);
}

void PadLowering::LoweringINT8(PatternRewriter &rewriter,
                               top::PadOp op, bool asymmetric) const {
  int64_t in_zp;
  double in_scale;
  Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);

  std::vector<NamedAttribute> attrs;
  auto val = op.val().convertToDouble();
  val = std::round(val / in_scale + in_zp);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.paddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  attrs.push_back(rewriter.getNamedAttr("mode", op.modeAttr()));

  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, op->getOperands(),
                                          attrs);
}

void PadLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::PadOp op) const {
  lowering_common_float<tpu::PadOp, BFloat16Type>(rewriter, op);
}

void PadLowering::LoweringF16(PatternRewriter &rewriter,
                              top::PadOp op) const {
  lowering_common_float<tpu::PadOp, Float16Type>(rewriter, op);
}

void PadLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::PadOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
