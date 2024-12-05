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
class SliceCastSwapPattern : public OpRewriterPatternEx<tpu::SliceOp> {
public:
  SliceCastSwapPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::SliceOp>(context, "SliceCastSwapPattern") {}

  LogicalResult matchAndRewriteImpl(tpu::SliceOp op,
                                    PatternRewriter &rewriter) const override {

    auto input_op =
        dyn_cast_or_null<tpu::CastOp>(op.getInput().getDefiningOp());
    if (!op.getResult().hasOneUse()) {
      return failure();
    }
    auto output_op =
        dyn_cast_or_null<tpu::CastOp>(*op.getResult().getUsers().begin());
    if (!input_op || !output_op) {
      return failure();
    }
    auto in_input = input_op.getInput();
    auto out_output = output_op.getOutput();
    auto in_stype = module::getStorageType(in_input);
    auto out_stype = module::getStorageType(out_output);
    if (!in_stype.isInteger(32) || !out_stype.isInteger(32)) {
      return failure();
    }

    auto slice_out_shape = module::getShape(output_op.getInput());
    auto cast_shape = module::getShape(input_op.getInput());
    module::setShape(output_op.getOutput(), cast_shape);

    op.getOutput().replaceAllUsesWith(input_op.getOutput());

    auto after_cast = output_op.getOutput();
    rewriter.setInsertionPointAfterValue(after_cast);
    auto slice_type = module::getTypeLike(after_cast, cast_shape);
    auto loc = output_op.getLoc();
    module::setLocSuffix(output_op, "_new");
    auto new_slice_op = rewriter.create<tpu::SliceOp>(
        loc, slice_type,
        ValueRange{after_cast, op->getOperand(1), op->getOperand(2),
                   op->getOperand(3), op->getOperand(4)},
        op->getAttrs());

    module::setShape(new_slice_op.getOutput(), slice_out_shape);
    after_cast.replaceAllUsesExcept(new_slice_op.getOutput(), new_slice_op);

    rewriter.eraseOp(op);
    return success();
  }
  bool shouldPrint(tpu::SliceOp op) const override { return false; }
};

void tpu::SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SliceCastSwapPattern>(context);
}

} // namespace tpu
} // namespace tpu_mlir
