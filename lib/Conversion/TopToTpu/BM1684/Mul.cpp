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

void MulLowering::LoweringF32(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_float<tpu::MulOp, Float32Type>(rewriter, op);
}

void MulLowering::LoweringINT8(PatternRewriter &rewriter, top::MulOp op,
                               bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684
} // namespace tpu_mlir
