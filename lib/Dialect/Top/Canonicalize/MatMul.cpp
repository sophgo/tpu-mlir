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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

// MatMul + Add(weight) => MatMul
struct MatMulWithBias : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    if (module::isWeight(filter) == false) {
      return failure();
    }
    if (module::isNone(op.getBias()) == false) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }
    auto user = *op->getUsers().begin();
    auto add_op = dyn_cast<AddOp>(user);
    if (!add_op) {
      return failure();
    }
    if (add_op.getNumOperands() != 2) {
      return failure();
    }
    Value bias = nullptr;
    bool bias_is_weight = false;
    for (auto v : add_op.getOperands()) {
      if (module::isWeight(v)) {
        bias = v;
        bias_is_weight = true;
        break;
      }
    }
    if (!bias_is_weight) {
      return failure();
    }
    auto p = op.parseParam();
    if (p.batch > 1) {
      // TODO: not support batch matmul; need to support
      return failure();
    }
    if (module::getNumElements(bias) != p.N) {
      // TODO: maybe == 1 is OK
      return failure();
    }
    auto bias_op = bias.getDefiningOp();
    if (!bias_op->isBeforeInBlock(op)) {
      bias_op->moveBefore(op);
    }
    op->setOperand(2, bias);
    op->setLoc(add_op.getLoc());
    add_op.replaceAllUsesWith(op.getOperation());
    rewriter.eraseOp(add_op);
    return success();
  }
};

// merge n and c if c is small and n is large
struct OptMatMulSmallCdim : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto left = op.getInput();
    auto right = op.getRight();
    auto out = op.getResult();

    auto mul_op = dyn_cast<MulConstOp>(left.getDefiningOp());
    if (!(mul_op && mul_op->hasOneUse())) {
      return failure();
    }
    auto mul_in = mul_op.getInput();
    auto mul_shape = module::getShape(mul_in);
    if (!mul_in.hasOneUse() || mul_shape.size() != 4 ||
        (mul_shape[1] >= mul_shape[2])) {
      return failure();
    }

    // 1. add ReshapeOp before MulConst
    rewriter.setInsertionPoint(mul_op);
    std::vector<int64_t> new_lshape(
        {mul_shape[0] * mul_shape[1], mul_shape[2], mul_shape[3]});
    auto type0 = RankedTensorType::get(new_lshape, rewriter.getF32Type());
    std::string in_name = module::getName(mul_in).str();
    std::string reshape0_name = in_name + "_reshape_left";
    auto loc0 = NameLoc::get(rewriter.getStringAttr(reshape0_name));
    auto reshape0_op =
        rewriter.create<ReshapeOp>(loc0, type0, ValueRange{mul_in});
    // mul_op->setOperand(0, reshape0_op);
    left.setType(type0);
    mul_in.replaceAllUsesExcept(reshape0_op.getOutput(), reshape0_op);

    // 2. add ReshapeOp before Right_in
    rewriter.setInsertionPoint(reshape0_op);
    auto rshape = module::getShape(right);
    std::vector<int64_t> new_rshape(
        {rshape[0] * rshape[1], rshape[2], rshape[3]});
    auto type1 = RankedTensorType::get(new_rshape, rewriter.getF32Type());
    std::string right_in_name = module::getName(right).str();
    std::string reshape1_name = right_in_name + "_reshape_right";
    auto loc1 = NameLoc::get(rewriter.getStringAttr(reshape1_name));
    auto reshape1_op =
        rewriter.create<ReshapeOp>(loc1, type1, ValueRange{right});
    // op->setOperand(1, reshape1_op);
    right.replaceAllUsesExcept(reshape1_op.getOutput(), reshape1_op);

    // 3. add ReshapeOp after MatMul out
    rewriter.setInsertionPointAfterValue(out);
    auto oshape = module::getShape(out);
    std::vector<int64_t> new_oshape(
        {oshape[0] * oshape[1], oshape[2], oshape[3]});
    auto type2 = RankedTensorType::get(new_oshape, rewriter.getF32Type());
    std::string out_name = module::getName(right).str();
    std::string reshape2_name = out_name + "_reshape_matmul";
    auto loc2 = NameLoc::get(rewriter.getStringAttr(reshape2_name));
    auto reshape2_op =
        rewriter.create<ReshapeOp>(loc2, out.getType(), ValueRange{out});
    out.setType(type2);
    out.replaceAllUsesExcept(reshape2_op.getOutput(), reshape2_op);

    return success();
  }
};

// Add Reshape op after non-keepdims MatMul to make layergroup easier
struct NoKeepDimsAddReshape : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getKeepDims()) {
      return failure();
    }

    // specail case for bert
    auto output = op.getResult();
    for (auto user : output.getUsers()) {
      if (auto packop = dyn_cast<top::PackOp>(user)) {
        return failure();
      }
    }

    // cache the output type and loc
    auto reshape_out = output.getType();
    auto out_loc = output.getLoc();

    // change the MatMul op into keepdims and recalculate the output shape
    op.setKeepDims(true);
    output.setType(UnrankedTensorType::get(module::getElementType(output)));
    output.setLoc(NameLoc::get(rewriter.getStringAttr(module::getName(output).str() + "_keepdims")));
    op.shape_inference();

    // add reshape op after Matmul
    rewriter.setInsertionPointAfter(op);
    auto reshape_op = rewriter.create<ReshapeOp>(out_loc, reshape_out, ValueRange{output});
    output.replaceAllUsesExcept(reshape_op.getOutput(), reshape_op);

    return success();
  }
};

void MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<MatMulWithBias, NoKeepDimsAddReshape>(context);
}
