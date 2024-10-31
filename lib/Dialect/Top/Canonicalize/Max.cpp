//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

struct MaxToMaxConst : public OpRewriterPatternEx<MaxOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MaxToMaxConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MaxOp>(context, "MaxToMaxConst") {}

  LogicalResult matchAndRewriteImpl(MaxOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }

    int left_elt_num = module::getNumElements(op.getInputs()[0]);
    int right_elt_num = module::getNumElements(op.getInputs()[1]);
    if (left_elt_num > 1 && right_elt_num > 1) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32())
      return failure();

    Value new_input;
    std::shared_ptr<std::vector<float>> const_val;
    bool weight_flag = false;
    if (left_elt_num == 1) {
      if (auto left_op =
              dyn_cast<WeightOp>(op.getInputs()[0].getDefiningOp())) {
        weight_flag = true;
        const_val = left_op.read_as_float();
      }
      new_input = op.getInputs()[1];
    }
    if (!weight_flag && right_elt_num == 1) {
      if (auto right_op =
              dyn_cast<WeightOp>(op.getInputs()[1].getDefiningOp())) {
        weight_flag = true;
        const_val = right_op.read_as_float();
      }
      new_input = op.getInputs()[0];
    } else {
      return failure();
    }

    if (!weight_flag)
      return failure();
    Type output = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val->at(0))));
    rewriter.replaceOpWithNewOp<MaxConstOp>(op, output, new_input, attrs);
    return success();
  }
};

void MaxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MaxToMaxConst>(context);
}
