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
  auto log_op = rewriter.create<tpu::ActiveOp>(
      log_loc, type, ValueRange{op.getInput()}, attrs);
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

void PowLowering::LoweringINT8(PatternRewriter &rewriter, top::PowOp op,
                               bool asymmetric) const {
  double g_ex = 0;
  g_ex = op.getExponent().convertToDouble();
  auto table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [g_ex](double val) { return std::pow(val, g_ex); }, 32);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684
} // namespace tpu_mlir
