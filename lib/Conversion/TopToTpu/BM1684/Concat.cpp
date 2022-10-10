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

void ConcatLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::ConcatOp op) const {
  lowering_common_float<tpu::ConcatOp>(rewriter, op);
}

void ConcatLowering::LoweringINT8(PatternRewriter &rewriter, top::ConcatOp op,
                                  bool asymmetric) const {
  lowering_common_int8<tpu::ConcatOp>(rewriter, op, false);
}

} // namespace bm1684
} // namespace tpu_mlir
