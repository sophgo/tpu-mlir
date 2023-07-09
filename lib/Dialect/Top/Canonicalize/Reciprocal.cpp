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

using namespace tpu_mlir::top;

// From: pow -> reduce_mean -> addconst -> sqrt -> reciprocal -> mul -> mul_w
//        |_______________________________________________________↑
// To: RMSNorm
struct MergeToRMSNormPattern : public OpRewritePattern<ReciprocalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReciprocalOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getConstVal().convertToDouble() != 1.0 || op.getDoRelu()) {
      return failure();
    }

    // check input of ReciprocalOp
    auto sqrt_op = dyn_cast<SqrtOp>(op.getInput().getDefiningOp());
    if (!sqrt_op) {
      return failure();
    }
    auto add_const_op =
        dyn_cast<AddConstOp>(sqrt_op.getInput().getDefiningOp());
    if (!add_const_op) {
      return failure();
    }
    auto eps = add_const_op.getConstVal().convertToDouble();
    if (eps <= 0) {
      return failure();
    }
    auto reduce_op =
        dyn_cast<ReduceOp>(add_const_op.getInput().getDefiningOp());
    if (!reduce_op) {
      return failure();
    }
    auto axes = module::getI64Array(reduce_op.getAxes());
    auto dim_size = module::getShape(reduce_op.getInput()).size();
    auto axis = axes->at(0);
    if (axis < 0) {
      axis += dim_size;
    }
    // RMSNorm only normalizes the last dimension
    if (!reduce_op || reduce_op.getMode().str() != "ReduceMean" ||
        axes->size() != 1 || axis != dim_size - 1) {
      return failure();
    }
    // pow op is transformed to mul in canonicalize pass
    auto pow_op = dyn_cast<MulOp>(reduce_op.getInput().getDefiningOp());
    if (!pow_op || pow_op.getNumOperands() != 2 ||
        pow_op.getOperand(0) != pow_op.getOperand(1) || pow_op.getDoRelu()) {
      return failure();
    }

    // check output of ReciprocalOp
    auto mul_op = dyn_cast_or_null<MulOp>(module::getNextOp(op));
    if (!mul_op || mul_op.getDoRelu() ||
        (mul_op.getOperand(0) != pow_op.getOperand(0) &&
         (mul_op.getOperand(1) != pow_op.getOperand(0)))) {
      return failure();
    }
    auto weight_mul_op = dyn_cast_or_null<MulOp>(module::getNextOp(mul_op));
    if (!weight_mul_op) {
      return failure();
    }
    auto operand0 = weight_mul_op.getOperand(0).getDefiningOp();
    auto operand1 = weight_mul_op.getOperand(1).getDefiningOp();
    auto weight = isa<WeightOp>(operand0) ? dyn_cast<WeightOp>(operand0)
                                          : dyn_cast<WeightOp>(operand1);

    if (!weight_mul_op || mul_op.getDoRelu() ||
        weight_mul_op.getNumOperands() != 2 || !weight) {
      return failure();
    }

    std::vector<NamedAttribute> attrs;
    auto eps_attr = rewriter.getNamedAttr("eps", rewriter.getF64FloatAttr(eps));
    attrs.push_back(eps_attr);
    rewriter.setInsertionPointAfter(weight_mul_op);
    rewriter.replaceOpWithNewOp<top::RMSNormOp>(
        weight_mul_op, weight_mul_op.getOutput().getType(),
        ValueRange{pow_op.getOperand(0), weight.getResult()}, attrs);
    return success();
  }
};

void ReciprocalOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<MergeToRMSNormPattern>(context);
}
