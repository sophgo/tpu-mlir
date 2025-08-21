//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void SelectiveScanLowering::LoweringF32(PatternRewriter &rewriter,
                                        top::SelectiveScanOp op) const {
  lowering_common_f32<tpu::SelectiveScanOp>(rewriter, op);
}

void SelectiveScanLowering::LoweringBF16(PatternRewriter &rewriter,
                                         top::SelectiveScanOp op) const {
  lowering_common_bf16<tpu::SelectiveScanOp>(rewriter, op);
}

void SelectiveScanLowering::LoweringF16(PatternRewriter &rewriter,
                                        top::SelectiveScanOp op) const {
  lowering_common_f16<tpu::SelectiveScanOp>(rewriter, op);
}

void SelectiveScanLowering::LoweringINT8(PatternRewriter &rewriter,
                                         top::SelectiveScanOp op,
                                         bool asymmetric) const {
  lowering_common_f16<tpu::SelectiveScanOp>(rewriter, op);
}

void SelectiveScanLowering::LoweringINT4(PatternRewriter &rewriter,
                                         top::SelectiveScanOp op,
                                         bool asymmetric) const {
  lowering_common_f16<tpu::SelectiveScanOp>(rewriter, op);
}

void SelectiveScanLowering::LoweringF8(PatternRewriter &rewriter,
                                       top::SelectiveScanOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SelectiveScanLowering::LoweringQuantized(PatternRewriter &rewriter,
                                              top::SelectiveScanOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
