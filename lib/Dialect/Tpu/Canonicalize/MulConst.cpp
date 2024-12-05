//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {
// case1: Fuse multiple mulconst ops into one
// only when in_dtype == out_dtype or in_dtype == fp8
class FuseMultiMulConst : public OpRewriterPatternEx<tpu::MulConstOp> {
public:
  FuseMultiMulConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MulConstOp>(context, "FuseMultiMulConst") {}

  LogicalResult matchAndRewriteImpl(tpu::MulConstOp op,
                                    PatternRewriter &rewriter) const override {

    // starts from the last mulconst op
    if (!op->hasOneUse() ||
        dyn_cast<tpu::MulConstOp>(module::getNextOp(op, 0))) {
      return failure();
    }

    auto input = op.getInput();
    auto in_dtype = BM168x::getDataType(input);
    auto out_dtype = BM168x::getDataType(op.getOutput());
    auto final_const_val = op.getConstVal().convertToDouble();
    auto prev_op = dyn_cast<tpu::MulConstOp>(input.getDefiningOp());
    if (!prev_op) {
      return failure();
    }

    while (in_dtype == out_dtype || in_dtype == DTYPE_F8E4M3 ||
           in_dtype == DTYPE_F8E5M2) {
      final_const_val *= prev_op.getConstVal().convertToDouble();
      input = prev_op.getInput();
      prev_op = dyn_cast<tpu::MulConstOp>(input.getDefiningOp());
      if (!prev_op) {
        break;
      }
      out_dtype = in_dtype;
      in_dtype = BM168x::getDataType(prev_op.getInput());
    }
    op.setConstValAttr(rewriter.getF64FloatAttr(final_const_val));
    op.setOperand(input);

    return success();
  }
  bool shouldPrint(tpu::MulConstOp op) const override { return false; }
};

// case2: Fuse cast to FP8 MulConst
class FuseCastToF8MulConst : public OpRewriterPatternEx<tpu::MulConstOp> {
public:
  FuseCastToF8MulConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MulConstOp>(context, "FuseCastToF8MulConst") {}

  LogicalResult matchAndRewriteImpl(tpu::MulConstOp op,
                                    PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto in_dtype = BM168x::getDataType(input);
    if (!(in_dtype == DTYPE_F8E4M3 || in_dtype == DTYPE_F8E5M2)) {
      return failure();
    }

    if (!op->hasOneUse()) {
      return failure();
    }

    auto castOp = dyn_cast<tpu::CastOp>(module::getNextOp(op, 0));
    if (!castOp) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tpu::MulConstOp>(
        castOp, castOp.getType(), ValueRange{input}, op->getAttrs());
    return success();
  }
  bool shouldPrint(tpu::MulConstOp op) const override { return false; }
};

void tpu::MulConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<FuseMultiMulConst, FuseCastToF8MulConst>(context);
}

} // namespace tpu
} // namespace tpu_mlir
