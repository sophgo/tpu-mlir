//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {

void ClipLowering::LoweringINT8(PatternRewriter &rewriter, top::ClipOp ClipOp,
                                bool asymmetric) const {
  lowering_common_bf16<tpu::ClipOp>(rewriter, ClipOp);
}

void ClipLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ClipOp ClipOp) const {
  lowering_common_bf16<tpu::ClipOp>(rewriter, ClipOp);
}
} // namespace cv18xx
} // namespace tpu_mlir
