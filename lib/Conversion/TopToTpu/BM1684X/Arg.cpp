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

void LoweringArg(PatternRewriter &rewriter, top::ArgOp op) {
  auto shape = module::getShape(op.getOutput());
  auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
  lowering_common<tpu::ArgOp>(rewriter, op, new_type);
  return;
}

void ArgLowering::LoweringF32(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringArg(rewriter, op);
}
void ArgLowering::LoweringINT4(PatternRewriter &rewriter, top::ArgOp op,
                               bool asymmetric) const {
  LoweringArg(rewriter, op);
}
void ArgLowering::LoweringINT8(PatternRewriter &rewriter, top::ArgOp op,
                               bool asymmetric) const {
  LoweringArg(rewriter, op);
}

void ArgLowering::LoweringBF16(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringArg(rewriter, op);
}

void ArgLowering::LoweringF16(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringArg(rewriter, op);
}

void ArgLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ArgOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
