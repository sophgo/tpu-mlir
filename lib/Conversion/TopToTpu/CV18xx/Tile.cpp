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

// todo num_dims > 4, do loop tile
void TileLowering::LoweringINT8(PatternRewriter &rewriter, top::TileOp TileOp,
                                bool asymmetric) const {
  lowering_common_int8<tpu::TileOp>(rewriter, TileOp.getOperation(),
                                    asymmetric, 2);
}

void TileLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TileOp TileOp) const {
  lowering_common_bf16<tpu::TileOp>(rewriter, TileOp.getOperation(), 2);
}

} // namespace cv18xx
} // namespace tpu_mlir
