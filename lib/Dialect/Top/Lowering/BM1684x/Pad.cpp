//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void top::PadOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                       bool asymmetric) {
  auto op = getOperation();
  int64_t in_zp;
  double in_scale;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, asymmetric);

  std::vector<NamedAttribute> attrs;
  auto val_ = val().convertToDouble();
  val_ = std::round(val_ / in_scale + in_zp);
  attrs.push_back(rewriter.getNamedAttr("paddings", paddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val_)));
  attrs.push_back(rewriter.getNamedAttr("mode", modeAttr()));

  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, op->getOperands(),
                                          attrs);
}

void top::PadOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::PadOp>(rewriter, getOperation());
}

void top::PadOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::PadOp, BFloat16Type>(rewriter, getOperation());
}

void top::PadOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::PadOp, Float16Type>(rewriter, getOperation());
}

void top::PadOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("not support now");
}
