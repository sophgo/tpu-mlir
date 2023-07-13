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

// y = x ^ n = e ^ (n * log(x))
// TODO: dangerous as need x > 0
void PowLowering::LoweringF32(PatternRewriter &rewriter, top::PowOp op) const {
  auto name = module::getName(op.getOutput());
  auto type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);
  auto log_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_log"));
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::LN)));
  auto log_op = rewriter.create<tpu::ActiveOp>(log_loc, type,
                                               ValueRange{op.getInput()}, attrs);
  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr("const_val", op.getExponentAttr()));
  auto mul_op = rewriter.create<tpu::MulConstOp>(
      mul_loc, type, ValueRange{log_op.getOutput()}, attrs);
  auto ex_loc = op.getLoc();
  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
  auto ex_op = rewriter.create<tpu::ActiveOp>(
      ex_loc, type, ValueRange{mul_op.getOutput()}, attrs);
  op.replaceAllUsesWith(ex_op.getOperation());
  rewriter.eraseOp(op);
}

static double g_ex = 0;
void PowLowering::LoweringINT8(PatternRewriter &rewriter, top::PowOp op,
                               bool asymmetric) const {
  g_ex = op.getExponent().convertToDouble();
  auto table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                          [](double val) { return std::pow(val, g_ex); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}
void PowLowering::LoweringINT4(PatternRewriter &rewriter, top::PowOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void PowLowering::LoweringBF16(PatternRewriter &rewriter, top::PowOp op) const {
  LoweringF32(rewriter, op);
}

void PowLowering::LoweringF16(PatternRewriter &rewriter, top::PowOp op) const {
  LoweringF32(rewriter, op);
}

void PowLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::PowOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
