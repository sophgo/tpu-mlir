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

void RemainderLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::RemainderOp op) const {

  auto name = module::getName(op.getOutput());
  auto type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);

  std::vector<NamedAttribute> attrs;
  // TODO : use mulconst / mul when divisor is weight
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
  std::vector<Value> op_values;
  op_values.push_back(op->getOperand(1));
  op_values.push_back(floor_op.getOutput());

  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  auto mul_op = rewriter.create<tpu::MulOp>(mul_loc, type, op_values, attrs);

  // sub
  op_values.clear();
  op_values.push_back(op->getOperand(0));
  op_values.push_back(mul_op.getOutput());

  auto sub_loc = op.getLoc();
  auto sub_op = rewriter.create<tpu::SubOp>(sub_loc, type, op_values, attrs);

  op.replaceAllUsesWith(sub_op.getOperation());
  rewriter.eraseOp(op);
}

void RemainderLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::RemainderOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
