//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void PadLowering::LoweringF32(PatternRewriter &rewriter, top::PadOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.getOperand());
  auto noneOp = module::getNoneOp(op);
  for (int i = op->getNumOperands(); i < 4; i++) {
    operands.push_back(noneOp);
  }

  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, op.getOutput().getType(),
                                          operands, op->getAttrs());
}

void PadLowering::LoweringINT8(PatternRewriter &rewriter, top::PadOp op,
                               bool asymmetric) const {
  int64_t in_zp;
  double in_scale;
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  operands.push_back(module::getNoneOp(op));
  operands.push_back(module::getNoneOp(op));
  operands.push_back(module::getNoneOp(op));
  module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);

  std::vector<NamedAttribute> attrs;
  auto val = op.getVal().convertToDouble();
  val = std::round(val / in_scale + in_zp);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.getPaddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  attrs.push_back(rewriter.getNamedAttr("mode", op.getModeAttr()));

  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands, attrs);
}

} // namespace bm1684
} // namespace tpu_mlir
