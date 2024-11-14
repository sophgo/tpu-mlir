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
void CustomLowering::LoweringINT8(PatternRewriter &rewriter, top::CustomOp op,
                                  bool asymmetric) const {
  lowering_common_int8<tpu::CustomOp>(rewriter, op, asymmetric);
}

void CustomLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::CustomOp op) const {
  lowering_common_bf16<tpu::CustomOp>(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir
