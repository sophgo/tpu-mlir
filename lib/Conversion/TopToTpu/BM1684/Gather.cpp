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

void GatherLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::GatherOp op) const {
  rewriter.setInsertionPointAfter(op);
  if (module::isWeight(op.getInput())) {
    // need insert Weight2ActivationOp before Gather's input
    auto insert_loc =
        module::getLocLike(op.getInput(), "convert_to_activation");
    auto weight2activation_op = rewriter.create<tpu::Weight2ActivationOp>(
        insert_loc, op.getInput().getType(), ValueRange{op.getInput()});

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
    auto new_gather_op = rewriter.create<tpu::GatherOp>(
        op.getLoc(), op.getOutput().getType(),
        ValueRange{weight2activation_op.getOutput(), op.getIndices(),
                   module::getNoneOp(op)},
        attrs);
    op.getResult().replaceAllUsesWith(new_gather_op.getOutput());
    rewriter.eraseOp(op);
    return;
  }
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (module::isWeight(op.getIndices())) {
    auto wOp = op.getIndices().getDefiningOp<top::WeightOp>();
    operands.push_back(wOp.clone_int(op));
  } else {
    operands.push_back(op.getIndices());
  }
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  rewriter.replaceOpWithNewOp<tpu::GatherOp>(op, op.getOutput().getType(),
                                             operands, op->getAttrs());
}

void GatherLowering::LoweringINT8(PatternRewriter &rewriter, top::GatherOp op,
                                  bool asymmetric) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
