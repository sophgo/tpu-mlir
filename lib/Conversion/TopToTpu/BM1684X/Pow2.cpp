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
// y = n ^ x = e ^ (x * log(n))
// TODO: dangerous as need x > 0
void Pow2Lowering::LoweringF32(PatternRewriter &rewriter,
                               top::Pow2Op op) const {
  auto name = module::getName(op.getOutput());
  auto type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);
  std::vector<NamedAttribute> attrs;
  auto n = std::log(op.getConstVal().convertToDouble());
  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  attrs.clear();
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(n)));
  auto mul_op = rewriter.create<tpu::MulConstOp>(
      mul_loc, type, ValueRange{op.getInput()}, attrs);
  auto ex_loc = op.getLoc();
  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
  auto ex_op = rewriter.create<tpu::ActiveOp>(
      ex_loc, type, ValueRange{mul_op.getOutput()}, attrs);
  op.replaceAllUsesWith(ex_op.getOperation());
  rewriter.eraseOp(op);
}

void Pow2Lowering::LoweringINT8(PatternRewriter &rewriter, top::Pow2Op op,
                                bool asymmetric) const {
  double g_x = op.getConstVal().convertToDouble();
  auto table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                          [g_x](double val) { return std::pow(g_x, val); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}
void Pow2Lowering::LoweringINT4(PatternRewriter &rewriter, top::Pow2Op op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void Pow2Lowering::LoweringBF16(PatternRewriter &rewriter,
                                top::Pow2Op op) const {
  if (module::isMARS3() || module::isSGTPUV8()) {
    auto name = module::getName(op.getOutput());
    auto type = getQuantFloatType<BFloat16Type>(op.getOutput());
    rewriter.setInsertionPointAfter(op);
    std::vector<NamedAttribute> attrs;
    auto n = std::log(op.getConstVal().convertToDouble());
    auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
    attrs.clear();
    attrs.push_back(
        rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(n)));
    auto mul_op = rewriter.create<tpu::MulConstOp>(
        mul_loc, type, ValueRange{op.getInput()}, attrs);
    auto ex_loc = op.getLoc();
    attrs.clear();
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
    auto ex_op = rewriter.create<tpu::ActiveOp>(
        ex_loc, type, ValueRange{mul_op.getOutput()}, attrs);
    op.replaceAllUsesWith(ex_op.getOperation());
    rewriter.eraseOp(op);
  } else {
    LoweringF32(rewriter, op);
  }
}

void Pow2Lowering::LoweringF16(PatternRewriter &rewriter,
                               top::Pow2Op op) const {
  LoweringF32(rewriter, op);
}

void Pow2Lowering::LoweringF8(PatternRewriter &rewriter, top::Pow2Op op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void Pow2Lowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::Pow2Op op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, true);
}
} // namespace bm1684x
} // namespace tpu_mlir
