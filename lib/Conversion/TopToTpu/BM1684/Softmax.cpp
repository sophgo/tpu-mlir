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

void SoftmaxLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SoftmaxOp op) const {
  lowering_common_f32<tpu::SoftmaxOp>(rewriter, op);
}

void SoftmaxLowering::LoweringINT8(PatternRewriter &rewriter, top::SoftmaxOp op,
                                   bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684
} // namespace tpu_mlir
