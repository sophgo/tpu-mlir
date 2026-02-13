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

void ConcatVolumeLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::ConcatVolumeOp op) const {
  lowering_common_f32<tpu::ConcatVolumeOp>(rewriter, op);
}

void ConcatVolumeLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::ConcatVolumeOp op,
                                        bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ConcatVolumeLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::ConcatVolumeOp op,
                                        bool asymmetric) const {
  lowering_common_int8<tpu::ConcatVolumeOp>(rewriter, op.getOperation(),
                                            asymmetric);
}

void ConcatVolumeLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::ConcatVolumeOp op) const {
  lowering_common_bf16<tpu::ConcatVolumeOp>(rewriter, op);
}

void ConcatVolumeLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::ConcatVolumeOp op) const {
  lowering_common_f16<tpu::ConcatVolumeOp>(rewriter, op);
}

void ConcatVolumeLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::ConcatVolumeOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ConcatVolumeLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::ConcatVolumeOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
