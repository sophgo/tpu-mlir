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

void top::MulOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                       bool asymmetric) {
  auto op = getOperation();
  const int nInputs = op->getNumOperands();
  if (asymmetric == false) {
    OpBuilder builder(op);
    std::vector<Value> operands;
    double scale;
    int64_t zp_o;
    double scale_o;
    Quant::getScaleAndZeroPoint(output(), scale_o, zp_o, asymmetric);

    double scale_i;
    int64_t zp;
    for (int i = 0; i < nInputs; i++) {
      auto input = op->getOperand(i);
      operands.push_back(input);
      Quant::getScaleAndZeroPoint(input, scale_i, zp, asymmetric);
      if (i == 0)
        scale = scale_i;
      else
        scale *= scale_i;
    }

    scale /= scale_o;

    int multiplier;
    int rshift;
    get_scale_and_shift(scale, multiplier, rshift, 8);

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("do_relu", do_reluAttr()));
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier", rewriter.getI64IntegerAttr(multiplier)));
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(rshift)));
    auto newType = Quant::getQuantInt8Type(output(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::MulOp>(op, newType, operands, attrs);
  } else {
    lowering_f32_bm1684x(rewriter);
  }
}

void top::MulOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MulOp, Float32Type>(rewriter, getOperation());
}

void top::MulOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MulOp, BFloat16Type>(rewriter, getOperation());
}

void top::MulOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MulOp, Float16Type>(rewriter, getOperation());
}

void top::MulOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  llvm_unreachable("Not Implemented");
}
