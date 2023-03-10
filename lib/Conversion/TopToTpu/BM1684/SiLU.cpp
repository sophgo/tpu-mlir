//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir{
namespace bm1684 {

void SiLULowering::LoweringF32(PatternRewriter &rewriter, top::SiLUOp op) const {
    // SiLU = x * sigmoid(x)
    auto o_name = module::getName(op.getOutput());
    std::vector<NamedAttribute> attrs;
    // sigmoid(x)
    auto sigmoid_loc = NameLoc::get(rewriter.getStringAttr(o_name.str() + "_sigmoid"));
    auto sigmoid_op = rewriter.create<tpu::ActiveOp>(sigmoid_loc,
                        op.getOutput().getType(), ValueRange{op.getInput()}, attrs);
    auto sigmoid_op_ = sigmoid_op.getOperation();
    sigmoid_op_ ->setAttr("mode", tpu::ActiveModeAttr::get(sigmoid_op.getContext(), tpu::ActiveMode::SIGMOID));
    // x
    attrs.clear();
    auto mul_op = rewriter.create<tpu::MulOp>(op.getLoc() ,
                    op.getOutput().getType(), ValueRange{op.getInput(), sigmoid_op.getOutput()}, attrs);
    op.replaceAllUsesWith(mul_op.getOperation());
    op.erase();
}

void SiLULowering::LoweringINT8(PatternRewriter &rewriter, top::SiLUOp op, bool asymmetric) const {
   llvm_unreachable("Now BM1684 Not Implemented Int8 Lowering");
}

} // namespace bm1684
} // namespace tpu_mlir
