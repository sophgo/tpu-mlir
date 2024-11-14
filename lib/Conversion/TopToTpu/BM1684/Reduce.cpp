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

void ReduceLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ReduceOp op) const {
  auto mode = op.getMode();
  if (mode != "ReduceL2") {
    lowering_common_f32<tpu::ReduceOp>(rewriter, op, 3);
  } else {
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    auto name = module::getName(op.getOutput());
    auto square_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_Square"));
    auto square_op = rewriter.create<tpu::ActiveOp>(
        square_loc, op.getInput().getType(), ValueRange{op.getInput()}, attrs);
    square_op->setAttr("mode", tpu::ActiveModeAttr::get(
                                   op.getContext(), tpu::ActiveMode::SQUARE));
    attrs.clear();
    auto reducesum_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_ReduceSum"));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceSum")));
    attrs.push_back(rewriter.getNamedAttr("axes", op.getAxes()));
    attrs.push_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    operands.push_back(square_op.getOutput());
    auto noneOp = module::getNoneOp(op);
    for (int i = op->getNumOperands(); i < 3; i++) {
      operands.push_back(noneOp);
    }
    auto reduce_op = rewriter.create<tpu::ReduceOp>(
        reducesum_loc, op.getOutput().getType(), operands, attrs);
    attrs.clear();
    auto sqrt_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_Sqrt"));
    auto sqrt_op = rewriter.create<tpu::ActiveOp>(
        sqrt_loc, op.getOutput().getType(), ValueRange{reduce_op.getOutput()},
        attrs);
    sqrt_op->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                      tpu::ActiveMode::SQRT));
    op.replaceAllUsesWith(sqrt_op.getOperation());
    rewriter.eraseOp(op);
  }
}

void ReduceLowering::LoweringINT8(PatternRewriter &rewriter, top::ReduceOp op,
                                  bool asymmetric) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
