//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {
struct AddConstZero : public OpRewriterPatternEx<tpu::AddConstOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  AddConstZero(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::AddConstOp>(context, "AddConstZero") {}

  LogicalResult matchAndRewriteImpl(tpu::AddConstOp op,
                                    PatternRewriter &rewriter) const override {
    bool is_type_match = module::getStorageType(op.getInput()) ==
                         module::getStorageType(op.getResult());
    bool is_identity = std::abs(op.getConstVal().convertToDouble()) < 1e-15 &&
                       op.getMultiplier() == 1 && op.getRshift() == 0;

    bool isTangents =
        module::isTrain() &&
        module::endsWith(module::getName(op.getResult()).str(), "_add_zero");
    if (!isTangents && is_type_match && is_identity) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    return failure();
  }
};

void tpu::AddConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<AddConstZero>(context);
}

} // namespace tpu
} // namespace tpu_mlir
