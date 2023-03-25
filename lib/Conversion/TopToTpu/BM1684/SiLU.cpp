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

// 1684's sigmoid nodechip will cover input, so here save input firstly.
// sigmoid's nodechip can be found nodechip_local_func_v2.c:316
void SiLULowering::LoweringF32(PatternRewriter &rewriter, top::SiLUOp op) const {
    auto op_ = op.getOperation();
    op_ ->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SILU));
    lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SiLULowering::LoweringINT8(PatternRewriter &rewriter, top::SiLUOp op, bool asymmetric) const {
   llvm_unreachable("Now BM1684 Not Implemented Int8 Lowering");
}

} // namespace bm1684
} // namespace tpu_mlir
