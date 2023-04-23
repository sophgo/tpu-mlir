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
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

struct MulToSiLU : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDoRelu() || op.getInputs().size() != 2) {
      return failure();
    }
    if (module::isUniformQuantized(op.getOutput()))
      return failure();
    auto in0_op = op.getInputs()[0].getDefiningOp();
    auto in1_op = op.getInputs()[1].getDefiningOp();
    Value in_value;
    SigmoidOp sigmoid_op = dyn_cast<SigmoidOp>(in1_op);
    if (sigmoid_op && sigmoid_op.getInput().getDefiningOp() == in0_op &&
        sigmoid_op->hasOneUse()) {
      in_value = op.getInputs()[0];
    } else if ((sigmoid_op = dyn_cast<SigmoidOp>(in0_op)) &&
               sigmoid_op.getInput().getDefiningOp() == in1_op &&
               sigmoid_op->hasOneUse()) {
      in_value = op.getInputs()[1];
    } else {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    rewriter.replaceOpWithNewOp<SiLUOp>(op, op.getResult().getType(),
                                        ValueRange{in_value}, attrs);
    return success();
  }
};

struct MulToMulConst : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (module::isUniformQuantized(op.getOutput())) {
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
    rewriter.replaceOpWithNewOp<MulConstOp>(op, output, new_input, attrs);
    return success();
  }
};

struct MulToScale : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }

    // check shape
    auto left_shape =
        op.getInputs()[0].getType().dyn_cast<TensorType>().getShape();
    auto right_shape =
        op.getInputs()[1].getType().dyn_cast<TensorType>().getShape();
    if (!(left_shape.size() == 4 && right_shape.size() == 4))
      return failure();
    if (left_shape[1] != right_shape[1])
      return failure();
    int left_elt_num = 1, right_elt_num = 1;
    for (int i = 0; i < left_shape.size(); ++i)
      left_elt_num *= left_shape[i];
    for (int i = 0; i < right_shape.size(); ++i)
      right_elt_num *= right_shape[i];
    if (left_elt_num != left_shape[1] && right_elt_num != right_shape[1])
      return failure();

    // Y = X * S + B
    Value X, S;
    if (left_elt_num == left_shape[1]) {
      X = op.getInputs()[1];
      S = op.getInputs()[0];
    } else if (right_elt_num == right_shape[1]) {
      X = op.getInputs()[0];
      S = op.getInputs()[1];
    } else {
      assert(0);
    }

    std::vector<float> scale(left_shape[1]);
    if (auto scale_ = dyn_cast<WeightOp>(S.getDefiningOp()))
      scale = *(scale_.read<float>());
    else
      return failure();

    auto scale_type =
        RankedTensorType::get({left_shape[1]}, rewriter.getF32Type());
    auto S_ = WeightOp::create(op, "scale", scale, scale_type);
    std::vector<float> bias(left_shape[1], 0);
    auto bias_type =
        RankedTensorType::get({left_shape[1]}, rewriter.getF32Type());
    auto B = WeightOp::create(op, "bias", bias, bias_type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "do_relu", op->getAttr("do_relu").cast<BoolAttr>()));
    attrs.push_back(rewriter.getNamedAttr(
        "relu_limit", op->getAttr("relu_limit").cast<FloatAttr>()));

    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getOutput().getType(),
                                         ValueRange{X, S_, B}, attrs);
    return success();
  }
};

//Mul + Mul
struct MulMerge : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32())
      return failure();
    auto a = op.getInputs()[0];
    auto b = op.getInputs()[1];
    if (module::isWeight(a)) {
      std::swap(a, b);
    } else if (module::isWeight(b)) {
      // do nothing
    } else {
      return failure();
    }
    auto mul = dyn_cast<MulOp>(a.getDefiningOp());
    if (!mul || mul.getInputs().size() != 2 || mul.getDoRelu()) {
      return failure();
    }
    auto c = mul.getInputs()[0];
    auto d = mul.getInputs()[1];
    if (module::isWeight(c)) {
      std::swap(c, d);
    } else if (module::isWeight(d)) {
      // do nothing
    } else {
      return failure();
    }
    auto b_op = b.getDefiningOp<WeightOp>();
    auto d_op = d.getDefiningOp<WeightOp>();
    auto b_data = b_op.read<float>();
    auto d_data = d_op.read<float>();
    auto b_shape = module::getShape(b);
    auto d_shape = module::getShape(d);
    std::vector<int64_t> o_shape;
    auto output =
        binary_mul(b_data->data(), d_data->data(), b_shape, d_shape, o_shape);
    auto type = RankedTensorType::get(o_shape, rewriter.getF32Type());
    rewriter.setInsertionPointAfter(op);
    auto weight = WeightOp::create<float>(op, "merged", *output, type);
    std::vector<NamedAttribute> attrs;
    if (op.getDoRelu()) {
      attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
      attrs.push_back(
          rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    }
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), ValueRange{c, weight},
                                       attrs);
    rewriter.eraseOp(mul);
    return success();
  }
};

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MulToSiLU, MulToMulConst, MulToScale, MulMerge>(context);
}
