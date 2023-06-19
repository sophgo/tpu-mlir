//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Conversion/TopToTpu/TopLowering.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

namespace tpu_mlir {
namespace bm1684x {

void NmsLowering::LoweringF32(PatternRewriter &rewriter, top::NmsOp op) const {
  lowering_common_f32<tpu::NmsOp>(rewriter, op);
}

void NmsLowering::LoweringINT4(PatternRewriter &rewriter, top::NmsOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void NmsLowering::LoweringINT8(PatternRewriter &rewriter, top::NmsOp op,
                               bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void NmsLowering::LoweringBF16(PatternRewriter &rewriter, top::NmsOp op) const {
  LoweringF32(rewriter, op);
}

void NmsLowering::LoweringF16(PatternRewriter &rewriter, top::NmsOp op) const {
  LoweringF32(rewriter, op);
}

void NmsLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::NmsOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
