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

void MulConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  lowering_common_f32<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MulConstOp op, bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
