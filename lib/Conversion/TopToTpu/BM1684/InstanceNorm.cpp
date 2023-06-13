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

void InstanceNormLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::InstanceNormOp op) const {
  auto op_ = op.getOperation();
  int channels = module::getShape(op.getInput())[1];
  op_->setAttr("num_groups", rewriter.getI64IntegerAttr(channels));
  lowering_common_f32<tpu::GroupNormOp>(rewriter, op_, 5);
}

void InstanceNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::InstanceNormOp op,
                                        bool asymmetric) const {
  // only support FP32
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
