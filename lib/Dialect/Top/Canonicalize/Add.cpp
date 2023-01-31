//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace mlir;
using namespace tpu_mlir::top;

struct AddToAddConst : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }

    if (op->hasAttr("coeff")) {
      auto coeffs =
          *module::getF64Array(op->getAttr("coeff").dyn_cast<ArrayAttr>());
      for (int i = 0; i < coeffs.size(); ++i) {
        if (coeffs.at(i) != double(0))
          return failure();
      }
    }

    auto left_shape =
        op.getInputs()[0].getType().dyn_cast<TensorType>().getShape();
    auto right_shape =
        op.getInputs()[1].getType().dyn_cast<TensorType>().getShape();
    int left_elt_num = 1, right_elt_num = 1;
    for (int i = 0; i < left_shape.size(); ++i)
      left_elt_num *= left_shape[i];
    for (int i = 0; i < right_shape.size(); ++i)
      right_elt_num *= right_shape[i];
    if (left_elt_num > 1 && right_elt_num > 1)
      return failure();

    Value new_input;
    std::shared_ptr<std::vector<float>> const_val;
    bool weight_flag = false;
    if (left_elt_num == 1) {
      if (auto left_op =
              dyn_cast<WeightOp>(op.getInputs()[0].getDefiningOp())) {
        weight_flag = true;
        const_val = left_op.read<float>();
      }
      new_input = op.getInputs()[1];
    } else if (right_elt_num == 1) {
      if (auto right_op =
              dyn_cast<WeightOp>(op.getInputs()[1].getDefiningOp())) {
        weight_flag = true;
        const_val = right_op.read<float>();
      }
      new_input = op.getInputs()[0];
    } else {
      assert(0);
    }

    if (!weight_flag)
      return failure();
    Type output = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val->at(0))));
    attrs.push_back(rewriter.getNamedAttr(
        "do_relu", op->getAttr("do_relu").cast<BoolAttr>()));
    attrs.push_back(rewriter.getNamedAttr(
        "relu_limit", op->getAttr("relu_limit").cast<FloatAttr>()));
    rewriter.replaceOpWithNewOp<AddConstOp>(op, output, new_input, attrs);
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<AddToAddConst>(context);
}