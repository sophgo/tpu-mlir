//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

struct SwapInput : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (!isa<WeightOp>(op.getInputs()[0].getDefiningOp()) &&
        !isa<WeightOp>(op.getInputs()[1].getDefiningOp())) {
      return failure();
    }
    auto coeffs = module::getF64Array(op.getCoeff(), 2, 1.0);
    for (auto c : *coeffs) {
      if (c != 1.0) {
        return failure();
      }
    }
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    if (isa<WeightOp>(lhs.getDefiningOp())) {
      op.setOperand(0, rhs);
      op.setOperand(1, lhs);
      return success();
    } else {
      return failure();
    }
  }
};

struct AddToScale : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (!isa<WeightOp>(op.getInputs()[1].getDefiningOp())) {
      return failure();
    }
    auto lhs_shape = module::getShape(op.getInputs()[0]);
    auto rhs_shape = module::getShape(op.getInputs()[1]);
    auto output_shape = module::getShape(op.getOutput());
    auto coeffs = module::getF64Array(op.getCoeff(), 2, 1.0);
    for (auto c : *coeffs) {
      if (c != 1.0) {
        return failure();
      }
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32()) {
      return failure();
    }

    if (output_shape.size() < 2 || lhs_shape.size() != rhs_shape.size() ||
        output_shape.size() - rhs_shape.size() > 1) {
      return failure();
    }

    if (rhs_shape[1] != lhs_shape[1]) {
      return failure();
    }

    auto elt_num = module::getNumElements(op.getInputs()[1]);
    if (elt_num != lhs_shape[1]) {
      return failure();
    }

    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    rewriter.setInsertionPoint(op);
    std::vector<float_t> weight_v(elt_num, 1.);
    auto rtype = RankedTensorType::get(rhs_shape.vec(), rewriter.getF32Type());
    auto w_scale =
        WeightOp::create(op.getOperation(), "_scale_weight", weight_v, rtype);
    operands.push_back(op.getInputs()[0]);
    operands.push_back(w_scale);
    operands.push_back(op.getInputs()[1]);
    attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
    attrs.push_back(rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getType(), operands, attrs);
    return success();
  }
};

struct AddToAddConst : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }

    auto coeffs = module::getF64Array(op.getCoeff(), 2, 1.0);
    for (auto c : *coeffs) {
      if (c != 1.0) {
        return failure();
      }
    }
    int left_elt_num = module::getNumElements(op.getInputs()[0]);
    int right_elt_num = module::getNumElements(op.getInputs()[1]);

    Value new_input;
    std::shared_ptr<std::vector<float>> const_val;
    bool weight_flag = false;
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32())
      return failure();
    if (left_elt_num == 1) {
      if (auto left_op =
              dyn_cast_or_null<WeightOp>(op.getInputs()[0].getDefiningOp())) {
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
      return failure();
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

// Add weight + Add Weight
struct AddMerge : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (module::isUniformQuantized(op.getOutput())) {
      return failure();
    }
    auto a = op.getInputs()[0];
    auto b = op.getInputs()[1];
    auto coeff0 = module::getF64Array(op.getCoeff(), 2, 1.0);
    auto a_coeff = coeff0->at(0);
    auto b_coeff = coeff0->at(1);
    if (module::isWeight(a)) {
      std::swap(a, b);
      std::swap(a_coeff, b_coeff);
    } else if (module::isWeight(b)) {
      // do nothing
    } else {
      return failure();
    }
    if (!a.hasOneUse()) {
      return failure();
    }
    auto add = dyn_cast<AddOp>(a.getDefiningOp());
    if (!add || add.getInputs().size() != 2 || add.getDoRelu()) {
      return failure();
    }
    auto c = add.getInputs()[0];
    auto d = add.getInputs()[1];
    auto coeff1 = module::getF64Array(add.getCoeff(), 2, 1.0);
    auto c_coeff = coeff1->at(0);
    auto d_coeff = coeff1->at(1);
    if (module::isWeight(c)) {
      std::swap(c, d);
      std::swap(c_coeff, d_coeff);
    } else if (module::isWeight(d)) {
      // do nothing
    } else {
      return failure();
    }
    auto b_op = b.getDefiningOp<WeightOp>();
    auto d_op = d.getDefiningOp<WeightOp>();
    auto b_data = b_op.read<float>();
    if (b_coeff != 1.0) {
      for (auto &b : *b_data) {
        b *= b_coeff;
      }
    }
    auto d_data = d_op.read<float>();
    if (d_coeff * a_coeff != 1.0) {
      for (auto &d : *d_data) {
        d *= d_coeff * a_coeff;
      }
    }
    auto b_shape = module::getShape(b);
    auto d_shape = module::getShape(d);
    std::vector<int64_t> o_shape;
    auto output =
        binary_add(b_data->data(), d_data->data(), b_shape, d_shape, o_shape);
    auto type = RankedTensorType::get(o_shape, rewriter.getF32Type());
    rewriter.setInsertionPointAfter(op);
    auto weight = WeightOp::create<float>(op, "merged", *output, type);
    auto coeff = a_coeff * c_coeff;
    std::vector<NamedAttribute> attrs;
    if (coeff != 1.0) {
      attrs.push_back(rewriter.getNamedAttr(
          "coeff", rewriter.getF64ArrayAttr({coeff, 1.0})));
    }
    if (op.getDoRelu()) {
      attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
      attrs.push_back(
          rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    }
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), ValueRange{c, weight},
                                       attrs);
    rewriter.eraseOp(add);
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<SwapInput, AddToAddConst, AddToScale, AddMerge>(context);
}
