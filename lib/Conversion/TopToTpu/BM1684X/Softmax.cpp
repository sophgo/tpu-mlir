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

void SoftmaxLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SoftmaxOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.input());
  auto none = Module::getNoneOp(op);
  operands.push_back(none);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(op, op.output().getType(),
                                              operands, attrs);
}

void SoftmaxLowering::LoweringINT8(PatternRewriter &rewriter, top::SoftmaxOp op,
                                   bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void SoftmaxLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SoftmaxOp op) const {
  LoweringF32(rewriter, op);
}

void SoftmaxLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SoftmaxOp op) const {
  LoweringF32(rewriter, op);
}

void SoftmaxLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::SoftmaxOp op) const {
  if (Quant::isUniformQuantized(op.input(), op.output()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  const int nInputs = op->getNumOperands();
  int64_t zeropoint;
  double i_scale;
  Quant::getScaleAndZeroPoint(op.input(), i_scale, zeropoint, true);
  std::vector<float> table(256, 0.0f);
  auto beta_v = op.beta().convertToDouble();
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
      op, op.output().getType(), ValueRange{op.input(), table_opd}, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
