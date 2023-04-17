//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"
#include "tpu_mlir/Support/LutFunc.h"

namespace tpu_mlir {
namespace bm1684 {

void AbsLowering::LoweringF32(PatternRewriter &rewriter, top::AbsOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::ABSVAL));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void AbsLowering::LoweringINT8(PatternRewriter &rewriter, top::AbsOp op,
                               bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684
} // namespace tpu_mlir
