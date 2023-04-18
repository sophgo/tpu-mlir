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

Value is_reshape_permute(Value in) {
  auto reshape0 = dyn_cast<ReshapeOp>(in.getDefiningOp());
  if (!reshape0 || !reshape0->hasOneUse()) {
    return NULL;
  }
  auto permute0 = dyn_cast<PermuteOp>(reshape0.getInput().getDefiningOp());
  if (!permute0 || !permute0->hasOneUse()) {
    return NULL;
  }
  auto reshape1 = dyn_cast<ReshapeOp>(permute0.getInput().getDefiningOp());
  if (!reshape1) {
    return permute0.getInput();
  } else if (!reshape1->hasOneUse()) {
    return NULL;
  } else {
    return reshape1.getInput();
  }
}

Value is_permute_reshape(Value in) {
  Value permute_out;
  auto reshape0 = dyn_cast<ReshapeOp>(in.getDefiningOp());
  if (!reshape0) {
    permute_out = in;
  } else if (!reshape0->hasOneUse()) {
    return NULL;
  } else {
    permute_out = reshape0.getInput();
  }
  auto permute0 = dyn_cast<PermuteOp>(permute_out.getDefiningOp());
  if (!permute0 || !permute0->hasOneUse())
    return NULL;
  auto reshape1 = dyn_cast<ReshapeOp>(permute0.getInput().getDefiningOp());
  if (!reshape1 || !reshape1->hasOneUse()) {
    return NULL;
  }
  return reshape1.getInput();
}

struct MatMul2Transformer : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    return failure();
    if (module::isWeight(filter) == false) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }
    Value matmul_out = is_reshape_permute(op.getInput());
    if (matmul_out == NULL) {
      return failure();
    }
    auto matmul0 = dyn_cast<MatMulOp>(matmul_out.getDefiningOp());
    if (!matmul0) {
      return failure();
    }
    auto softmax = dyn_cast<SoftmaxOp>(matmul0.getInput().getDefiningOp());
    if (!softmax || !softmax->hasOneUse()) {
      return failure();
    }
    Value mul_out;
    auto add = dyn_cast<AddOp>(softmax.getInput().getDefiningOp());
    if (!add) {
      mul_out = softmax.getInput();
    } else {
      mul_out = add.getInputs()[0];
    }
    auto mul_const = dyn_cast<MulConstOp>(mul_out.getDefiningOp());
    if (!mul_const || !mul_const->hasOneUse()) {
      return failure();
    }
    auto matmul1 = dyn_cast<MatMulOp>(mul_const.getInput().getDefiningOp());
    if (!matmul1) {
      return failure();
    }
    // queries
    Value matmul_out1 = is_permute_reshape(matmul1.getInput());
    if (matmul_out1 == NULL) {
      return failure();
    }
    auto matmul_queries = dyn_cast<MatMulOp>(matmul_out1.getDefiningOp());
    if (!matmul_queries || !module::isWeight(matmul_queries.getRight())) {
      return failure();
    }
    // keys
    auto permute0 = dyn_cast<PermuteOp>(matmul1.getRight().getDefiningOp());
    if (!permute0 || !permute0->hasOneUse())
      return failure();
    Value matmul_out2 = is_permute_reshape(permute0.getInput());
    if (matmul_out2 == NULL) {
      return failure();
    }
    auto matmul_keys = dyn_cast<MatMulOp>(matmul_out2.getDefiningOp());
    if (!matmul_keys || !module::isWeight(matmul_keys.getRight())) {
      return failure();
    }
    // values
    Value matmul_out3 = is_permute_reshape(matmul0.getRight());
    if (matmul_out3 == NULL) {
      return failure();
    }
    auto matmul_values = dyn_cast<MatMulOp>(matmul_out3.getDefiningOp());
    if (!matmul_values || !module::isWeight(matmul_values.getRight())) {
      return failure();
    }
    // if (matmul_keys.getInput() != matmul_queries.getInput() ||
    //     matmul_keys.getInput() != matmul_values.getInput()) {
    //   return failure();
    // }
    rewriter.setInsertionPointAfter(op);
    auto none = module::getNoneOp(op);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("scale", mul_const.getConstValAttr()));
    auto batch = module::getShape(op.getOutput())[0];
    auto shape = module::getShape(matmul0.getOutput());
    int64_t head;
    if (shape.size() == 3) {
      head = shape[0] / batch;
    } else {
      head = shape[1];
    }
    attrs.push_back(rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(head)));
    std::vector<Value> operands;
    operands.push_back(matmul_queries.getInput());
    operands.push_back(matmul_keys.getInput());
    operands.push_back(matmul_values.getInput());
    operands.push_back(matmul_queries.getRight());
    operands.push_back(matmul_queries.getBias());
    operands.push_back(matmul_keys.getRight());
    operands.push_back(matmul_keys.getBias());
    operands.push_back(matmul_values.getRight());
    operands.push_back(matmul_values.getBias());
    operands.push_back(op.getRight());
    operands.push_back(op.getBias());
    operands.push_back(add ? add.getInputs()[1] : none);
    auto transformer =
        rewriter.create<TransformerOp>(op.getLoc(), op.getOutput().getType(),
                                       operands, attrs);
    op.replaceAllUsesWith(transformer.getOperation());
    rewriter.eraseOp(op);
    return success();
  }
};

void MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<MatMulWithBias, MatMul2Transformer>(context);
}
