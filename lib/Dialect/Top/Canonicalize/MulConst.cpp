//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Support/Module.h"

using namespace mlir;
using namespace tpu_mlir::top;

struct RemoveMulConst : public OpRewritePattern<MulConstOp> {
  using OpRewritePattern::OpRewritePattern;
  RemoveMulConst(MLIRContext *context)
      : OpRewritePattern<MulConstOp>(context) {}
  LogicalResult matchAndRewrite(MulConstOp op,
                                PatternRewriter &rewriter) const override {
    //placeholder
    double const_val = op.getConstVal().convertToDouble();
    if (const_val != 1.0) {
      return failure();
    }
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

// merge into conv or matmul
struct MergeMulConst : public OpRewritePattern<MulConstOp> {
  using OpRewritePattern::OpRewritePattern;
  MergeMulConst(MLIRContext *context)
      : OpRewritePattern<MulConstOp>(context) {}
  LogicalResult matchAndRewrite(MulConstOp op,
                                PatternRewriter &rewriter) const override {
    double const_val = op.getConstVal().convertToDouble();
    bool do_relu = op.getDoRelu();
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }
    formerOp->dump();
    if (auto convOp = dyn_cast_or_null<top::ConvOp>(formerOp)) {
      if (convOp.getKernelShape().size() != 2) {
        return failure();
      }
      if (convOp.getDoRelu() == true && const_val < 0) {
        return failure();
      }
      auto weightOp = dyn_cast_or_null<top::WeightOp>(convOp.getFilter().getDefiningOp());
      auto biasOp = dyn_cast_or_null<top::WeightOp>(convOp.getBias().getDefiningOp());
      //This judge is for fix youdao bert bf16 acc issue.
      if (weightOp && biasOp) {
        auto weight_f32 = weightOp.read<float>();
        auto bias_f32 = biasOp.read<float>();
        if (weight_f32->size() == bias_f32->size()) {
          bool flag = true;
          for (uint i = 0; i < weight_f32->size(); i++) {
            if (weight_f32->at(i) + bias_f32->at(i) != 0) {
              flag = false;
              break;
            }
          }
          if (flag) {
            return failure();
          }
        }
      }
    } else if (auto fcOp = dyn_cast_or_null<top::MatMulOp>(formerOp)) {
      if ((fcOp.getDoRelu() && const_val < 0) ||
          !(isa<top::WeightOp>(fcOp.getRight().getDefiningOp()))) {
            return failure();
      }
    } else {
      return failure();
    }
    for (uint i = 0; i < formerOp->getNumOperands(); i++) {
      auto value = formerOp->getOperand(i);
      auto weightOp = dyn_cast_or_null<top::WeightOp>(value.getDefiningOp());
      if (!weightOp) {
        continue;
      }
      auto weight_f32 = weightOp.read<float>();
      //std::vector<float> new_weight_f32(weight_f32->size());
      for (auto &w : *weight_f32) {
        w *= const_val;
      }
      std::string weight_name = module::getName(value).str();
      auto weight_type = value.getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(weight_type.getShape(), rewriter.getF32Type());
      auto new_weight = top::WeightOp::create(formerOp, weight_name + "_mergeMulConst", *weight_f32, newType);
      formerOp->setOperand(i, new_weight);
    }
    formerOp->setLoc(op.getLoc());
    if (do_relu) {
      formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    }
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

void MulConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<RemoveMulConst, MergeMulConst>(context);
}
