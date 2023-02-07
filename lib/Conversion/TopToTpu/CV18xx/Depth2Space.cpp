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

#define DEBUG_TYPE "lowering-Depth2Space"

namespace tpu_mlir {
namespace cv18xx {

void Depth2SpaceLowering::LoweringINT8(PatternRewriter &rewriter, top::Depth2SpaceOp absOp,
                               bool asymmetric) const {
  lowering_common_int8<tpu::Depth2SpaceOp>(rewriter, absOp.getOperation(), asymmetric);
}

void Depth2SpaceLowering::LoweringBF16(PatternRewriter &rewriter,
                               top::Depth2SpaceOp absOp) const {
  lowering_common_bf16<tpu::Depth2SpaceOp>(rewriter, absOp.getOperation());
}

}
}
