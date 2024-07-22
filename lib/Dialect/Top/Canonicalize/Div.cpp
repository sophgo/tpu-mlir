//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct DivToMul : public OpRewriterPatternEx<DivOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  DivToMul(mlir::MLIRContext *context)
      : OpRewriterPatternEx<DivOp>(context, "DivToMul") {}

  LogicalResult matchAndRewriteImpl(DivOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }

    auto left = op.getInputs()[0];
    auto right = op.getInputs()[1];
    auto in_type = module::getStorageType(right);
    if (!in_type.isF32())
      return failure();
    auto right_shape = right.getType().dyn_cast<TensorType>().getShape();
    auto right_elt = 1;
    for (int i = 0; i < right_shape.size(); ++i)
      right_elt *= right_shape[i];

    std::vector<float> right_weight(right_elt);
    if (auto right_ = dyn_cast<WeightOp>(right.getDefiningOp())) {
      right_weight = *(right_.read_as_float());
      for (int i = 0; i < right_weight.size(); ++i) {
        right_weight[i] = float(1) / right_weight[i];
      }
    } else
      return failure();

    auto right_weight_ = WeightOp::create_float(op, "divisor", right_weight,
                                                right_shape, storage_type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "do_relu", op->getAttr("do_relu").cast<BoolAttr>()));
    attrs.push_back(rewriter.getNamedAttr(
        "relu_limit", op->getAttr("relu_limit").cast<FloatAttr>()));
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getOutput().getType(),
                                       ValueRange{left, right_weight_}, attrs);
    return success();
  }
};

struct DivToSoftSign : public OpRewriterPatternEx<DivOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  DivToSoftSign(mlir::MLIRContext *context)
      : OpRewriterPatternEx<DivOp>(context, "DivToSoftSign") {}

  LogicalResult matchAndRewriteImpl(DivOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto left = op.getOperands()[0];
    auto addConstOp = dyn_cast<AddConstOp>(op.getOperands()[1].getDefiningOp());
    if (!addConstOp || addConstOp.getConstVal().convertToDouble() != 1.0) {
      return failure();
    }

    auto absOp = dyn_cast<AbsOp>(addConstOp.getOperand().getDefiningOp());
    if (!absOp || left != absOp.getOperand()) {
      return failure();
    }

    std::vector<NamedAttribute> attrs;
    rewriter.replaceOpWithNewOp<SoftsignOp>(op, op.getOutput().getType(),
                                            ValueRange{left}, attrs);
    return success();
  }
};

void DivOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<DivToSoftSign, DivToMul>(context);
}
