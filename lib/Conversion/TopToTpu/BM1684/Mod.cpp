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

void ModLowering::LoweringF32(PatternRewriter &rewriter, top::ModOp op) const {

  auto name = module::getName(op.getOutput());
  auto type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);

  std::vector<NamedAttribute> attrs;
  // div
  auto div_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_div"));
  auto div_op = rewriter.create<tpu::DivOp>(div_loc, type,
                                            ValueRange{op.getInputs()}, attrs);

  // floor
  auto floor_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_floor"));
  attrs.push_back(rewriter.getNamedAttr(
      "mode",
      tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::FLOOR)));
  auto floor_op = rewriter.create<tpu::ActiveOp>(
      floor_loc, type, ValueRange(div_op.getOutput()), attrs);
  attrs.clear();

  // mul
  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  std::vector<Value> mul_inputs = {op->getOperand(1), floor_op.getOutput()};
  auto mul_op = rewriter.create<tpu::MulOp>(mul_loc, type, mul_inputs, attrs);

  // sub
  auto sub_loc = op.getLoc();
  std::vector<Value> sub_inputs = {op->getOperand(0), mul_op.getOutput()};
  auto sub_op = rewriter.create<tpu::SubOp>(sub_loc, type, sub_inputs, attrs);

  op.replaceAllUsesWith(sub_op.getOperation());
  rewriter.eraseOp(op);
}

void ModLowering::LoweringINT8(PatternRewriter &rewriter, top::ModOp op,
                               bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684
} // namespace tpu_mlir
