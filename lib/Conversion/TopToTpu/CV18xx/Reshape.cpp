//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-add"

namespace tpu_mlir {
namespace cv18xx {
void ReshapeLowering::LoweringINT8(PatternRewriter &rewriter, top::ReshapeOp op,
                                bool asymmetric) const {
  lowering_common_int8<tpu::ReshapeOp>(rewriter, op, asymmetric);
}

void ReshapeLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ReshapeOp op) const {
  lowering_common_bf16<tpu::ReshapeOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
