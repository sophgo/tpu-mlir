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

void LoweringArg(PatternRewriter &rewriter, top::ArgOp op, Type type) {
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  std::vector<Type> new_types;
  if (!module::isNone(op.getIndices())) {
    auto shape = module::getShape(op.getIndices());
    auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
    new_types.push_back(new_type);
  } else {
    new_types.push_back(op.getIndices().getType());
  }
  new_types.push_back(type);
  rewriter.replaceOpWithNewOp<tpu::ArgOp>(op, new_types, operands,
                                          op->getAttrs());
  return;
}

void ArgLowering::LoweringF32(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringArg(rewriter, op, getQuantFloatType(op.getValues()));
}

void ArgLowering::LoweringINT4(PatternRewriter &rewriter, top::ArgOp op,
                               bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ArgLowering::LoweringINT8(PatternRewriter &rewriter, top::ArgOp op,
                               bool asymmetric) const {
  if (asymmetric) {
    LoweringF16(rewriter, op);
  } else {
    LoweringArg(rewriter, op, getQuantInt8Type(op.getValues()));
  }
}

void ArgLowering::LoweringBF16(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringArg(rewriter, op, getQuantBF16Type(op.getValues()));
}

void ArgLowering::LoweringF16(PatternRewriter &rewriter, top::ArgOp op) const {
  LoweringArg(rewriter, op, getQuantF16Type(op.getValues()));
}

void ArgLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::ArgOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
