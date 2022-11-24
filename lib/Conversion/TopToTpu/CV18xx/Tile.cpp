//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

namespace tpu_mlir {
namespace cv18xx {

// todo num_dims > 4, do loop tile
void TileLowering::LoweringINT8(PatternRewriter &rewriter, top::TileOp TileOp,
                                bool asymmetric) const {
  lowering_common_int8<tpu::TileOp>(rewriter, TileOp.getOperation(),
                                    asymmetric);
}

void TileLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TileOp TileOp) const {
  lowering_common_bf16<tpu::TileOp>(rewriter, TileOp.getOperation());
}

} // namespace cv18xx
} // namespace tpu_mlir
