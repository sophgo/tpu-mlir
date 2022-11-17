//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {
void PadLowering::LoweringINT8(PatternRewriter &rewriter, top::PadOp op,
                                bool asymmetric) const {
  auto in_thr = Quant::getThreshold(op.input());
  auto in_scale = Quant::getScale(in_thr, true);
  std::vector<NamedAttribute> attrs;
  auto val = op.val().convertToDouble();
  val = Quant::to_int8(val / in_scale, ROUNDING_HALF_UP);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.paddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  attrs.push_back(rewriter.getNamedAttr("mode", op.modeAttr()));

  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, op->getOperands(),
                                          attrs);
}

void PadLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::PadOp op) const {
  lowering_common_bf16<tpu::PadOp>(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir
