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

void PermuteLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::PermuteOp op) const {
  lowering_common_f32<tpu::PermuteOp>(rewriter, op, 2);
}

void PermuteLowering::LoweringINT8(PatternRewriter &rewriter, top::PermuteOp op,
                                   bool asymmetric) const {
  if (asymmetric) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  lowering_common_int8<tpu::PermuteOp>(rewriter, op, asymmetric, 2);
}

} // namespace bm1684
} // namespace tpu_mlir
