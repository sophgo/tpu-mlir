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

void StridedSliceLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::StridedSliceOp op) const {
  lowering_common_f32<tpu::StridedSliceOp>(rewriter, op);
}

void StridedSliceLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::StridedSliceOp op,
                                        bool asymmetric) const {
  lowering_common_int8<tpu::StridedSliceOp>(rewriter, op, asymmetric);
}

} // namespace bm1684
} // namespace tpu_mlir
