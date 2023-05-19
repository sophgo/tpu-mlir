//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmcpu_common.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

static void Gather_lowring_common(PatternRewriter &rewriter, top::GatherOp op) {
  if (module::isBM1684Family() && module::isWeight(op.getInput())) {
    // need insert Weight2ActivationOp before Gather's input
    rewriter.setInsertionPointAfter(op);
    auto name = module::getName(op.getInput());
    auto insert_loc = NameLoc::get(
        rewriter.getStringAttr(name.str() + "_convert_to_activation"));
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
    op.erase();
  } else {
    lowering_common_f32<tpu::GatherOp>(rewriter, op, 3, 1);
  }
}

void GatherLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::GatherOp op) const {
  Gather_lowring_common(rewriter, op);
}

void GatherLowering::LoweringINT8(PatternRewriter &rewriter, top::GatherOp op,
                                  bool asymmetric) const {
  Gather_lowring_common(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
