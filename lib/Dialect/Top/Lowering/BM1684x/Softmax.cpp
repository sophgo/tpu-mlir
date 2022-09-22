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

void top::SoftmaxOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                           bool asymmetric) {
  lowering_f32_bm1684x(rewriter);
}

void top::SoftmaxOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  auto op = getOperation();
  std::vector<Value> operands;
  operands.push_back(input());
  auto none = Module::getNoneOp(op);
  operands.push_back(none);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(op, output().getType(), operands,
                                              attrs);
}

void top::SoftmaxOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  // lowering_common_float<tpu::SoftmaxOp, BFloat16Type>(rewriter,
  // getOperation());
  lowering_f32_bm1684x(rewriter);
}

void top::SoftmaxOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  // lowering_common_float<tpu::SoftmaxOp, Float16Type>(rewriter,
  // getOperation());
  lowering_f32_bm1684x(rewriter);
}

void top::SoftmaxOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  if (Quant::isUniformQuantized(input(), output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  auto op = getOperation();
  const int nInputs = op->getNumOperands();
  int64_t zeropoint;
  double i_scale;
  Quant::getScaleAndZeroPoint(input(), i_scale, zeropoint, true);
  std::vector<float> table(256, 0.0f);
  auto beta_v = beta().convertToDouble();
  auto scale = -i_scale * beta_v;
  for (int i = 0; i < 256; ++i) {
    table[i] = std::exp(scale * i);
  }
  auto table_opd = create_lookup_table(op, table);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(
      op, output().getType(), ValueRange{input(), table_opd}, attrs);
}
