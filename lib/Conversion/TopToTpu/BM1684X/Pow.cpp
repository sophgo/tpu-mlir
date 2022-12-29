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
  auto name = module::getName(op.output());
  auto type = op.output().getType();
  rewriter.setInsertionPointAfter(op);
  auto log_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_log"));
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::LN)));
  auto log_op = rewriter.create<tpu::ActiveOp>(log_loc, type,
                                               ValueRange{op.input()}, attrs);
  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr("const_val", op.exponentAttr()));
  auto mul_op = rewriter.create<tpu::MulConstOp>(
      mul_loc, type, ValueRange{log_op.output()}, attrs);
  auto ex_loc = op.getLoc();
  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
  auto ex_op = rewriter.create<tpu::ActiveOp>(
      ex_loc, type, ValueRange{mul_op.output()}, attrs);
  op.output().replaceAllUsesWith(ex_op.output());
}

static double g_ex = 0;
void PowLowering::LoweringINT8(PatternRewriter &rewriter, top::PowOp op,
                               bool asymmetric) const {
  auto stype = module::getStorageType(op.output());
  g_ex = op.exponent().convertToDouble();
  auto table =
      create_lookup_table(op.input(), op.output(), asymmetric,
                          [](double val) { return std::pow(val, g_ex); });
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table});
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
