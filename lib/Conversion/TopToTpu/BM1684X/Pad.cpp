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

static void LowerPadCommon(PatternRewriter &rewriter, top::PadOp op,
                           Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.input());
  operands.push_back(Module::getNoneOp(op));
  operands.push_back(Module::getNoneOp(op));
  auto output = op->getResult(0);
  Type newType;

  if (type.isF16()) {
    newType = getQuantF16Type(output);
  } else if (type.isBF16()) {
    newType = getQuantBF16Type(output);
  } else {
    newType = output.getType();
  }
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands,
                                             op->getAttrs());
}

void PadLowering::LoweringF32(PatternRewriter &rewriter, top::PadOp op) const {
  LowerPadCommon(rewriter, op, rewriter.getF32Type());
}

void PadLowering::LoweringINT8(PatternRewriter &rewriter, top::PadOp op,
                               bool asymmetric) const {
  int64_t in_zp;
  double in_scale;
  std::vector<Value> operands;
  operands.push_back(op.input());
  operands.push_back(Module::getNoneOp(op));
  operands.push_back(Module::getNoneOp(op));
  Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);

  std::vector<NamedAttribute> attrs;
  auto val = op.val().convertToDouble();
  val = std::round(val / in_scale + in_zp);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.paddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  attrs.push_back(rewriter.getNamedAttr("mode", op.modeAttr()));

  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands, attrs);
}

void PadLowering::LoweringBF16(PatternRewriter &rewriter, top::PadOp op) const {
  LowerPadCommon(rewriter, op, rewriter.getBF16Type());
}

void PadLowering::LoweringF16(PatternRewriter &rewriter, top::PadOp op) const {
  LowerPadCommon(rewriter, op, rewriter.getF16Type());
}

void PadLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::PadOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.input());
  operands.push_back(Module::getNoneOp(op));
  operands.push_back(Module::getNoneOp(op));
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, op.output().getType(),
                                          operands, op->getAttrs());
}

} // namespace bm1684x
} // namespace tpu_mlir
