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

void HardSwishLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::HardSwishOp op) const {
  // y=x*min(max(x+3,0), 3)/6
  auto name = module::getName(op.getOutput());
  rewriter.setInsertionPointAfter(op);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(3.f)));
  auto add_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_add"));
  auto add_op = rewriter.create<tpu::AddConstOp>(
      add_loc, op.getOutput().getType(), ValueRange{op.getInput()}, attrs);
  attrs.clear();
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(0.f)));
  auto max_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_max"));
  auto max_op = rewriter.create<tpu::MaxConstOp>(
      max_loc, op.getOutput().getType(), ValueRange{add_op.getOutput()}, attrs);
  attrs.clear();
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(6.f)));
  auto min_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_min"));
  auto min_op = rewriter.create<tpu::MinConstOp>(
      min_loc, op.getOutput().getType(), ValueRange{max_op.getOutput()}, attrs);
  attrs.clear();
  attrs.push_back(
      rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(1.0 / 6.0)));
  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  auto mul_op = rewriter.create<tpu::MulConstOp>(
      mul_loc, op.getOutput().getType(), ValueRange{min_op.getOutput()}, attrs);
  attrs.clear();
  auto mul2_op = rewriter.create<tpu::MulOp>(
      op.getLoc(), op.getOutput().getType(),
      ValueRange{op.getInput(), mul_op.getOutput()}, attrs);
  op.replaceAllUsesWith(mul2_op.getOperation());
  rewriter.eraseOp(op);
}

void HardSwishLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::HardSwishOp op,
                                     bool asymmetric) const {
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [](double x) { return x * std::min(std::max(x + 3, 0.), 6.) / 6.; }, 32);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684
} // namespace tpu_mlir
