//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-permute"
namespace tpu_mlir {
namespace cv18xx {
void PermuteLowering::LoweringINT8(PatternRewriter &rewriter, top::PermuteOp op,
                                   bool asymmetric) const {
  lowering_common_int8<tpu::PermuteOp>(rewriter, op, asymmetric, 2);
}

void PermuteLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::PermuteOp op) const {
  lowering_common_bf16<tpu::PermuteOp>(rewriter, op, 2);
}

} // namespace cv18xx
} // namespace tpu_mlir
