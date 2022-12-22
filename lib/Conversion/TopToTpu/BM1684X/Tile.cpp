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

void TileLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TileOp op) const {
  lowering_common_f32<tpu::TileOp>(rewriter, op);
}

void TileLowering::LoweringINT8(PatternRewriter &rewriter, top::TileOp op,
                                bool asymmetric) const {
  lowering_common_int8<tpu::TileOp>(rewriter, op, asymmetric);
}
void TileLowering::LoweringINT4(PatternRewriter &rewriter, top::TileOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void TileLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TileOp op) const {
  lowering_common_bf16<tpu::TileOp>(rewriter, op);
}

void TileLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TileOp op) const {
  lowering_common_f16<tpu::TileOp>(rewriter, op);
}

void TileLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::TileOp op) const {
  lowering_common<tpu::TileOp>(rewriter, op, op.output().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
