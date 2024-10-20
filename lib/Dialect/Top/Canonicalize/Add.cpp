//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct SwapInput : public OpRewriterPatternEx<AddOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  SwapInput(mlir::MLIRContext *context)
      : OpRewriterPatternEx<AddOp>(context, "SwapInput") {}

  LogicalResult matchAndRewriteImpl(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (!isa<WeightOp>(
            module::getOriValue(op.getInputs()[0]).getDefiningOp()) &&
        !isa<WeightOp>(
            module::getOriValue(op.getInputs()[1]).getDefiningOp())) {
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
    if (isa<WeightOp>(module::getOriValue(lhs).getDefiningOp())) {
      op.setOperand(0, rhs);
      op.setOperand(1, lhs);
      return success();
    } else {
      return failure();
    }
  }
};

struct AddToScale : public OpRewriterPatternEx<AddOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

    AddToScale(mlir::MLIRContext *context)
      : OpRewriterPatternEx<AddOp>(context, "AddToScale") {}

  LogicalResult matchAndRewriteImpl(AddOp op,
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

    if (lhs_shape == rhs_shape) {
      return failure();
    }
    for (auto c : *coeffs) {
      if (c != 1.0) {
        return failure();
      }
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
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
    auto w_scale = WeightOp::create_float(op.getOperation(), "_scale_weight", weight_v,
                                         rhs_shape, storage_type);
    operands.push_back(op.getInputs()[0]);
    operands.push_back(w_scale);
    operands.push_back(op.getInputs()[1]);
    attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
    attrs.push_back(rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getType(), operands, attrs);
    return success();
  }
};

struct AddToRope : public OpRewriterPatternEx<AddOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  AddToRope(mlir::MLIRContext *context)
      : OpRewriterPatternEx<AddOp>(context, "AddToRope") {}

  LogicalResult matchAndRewriteImpl(AddOp op, PatternRewriter &rewriter) const override {
    // Check the number of inputs
    if (op.getInputs().size() != 2) {
      return failure();
    }

    // Define weights and flags
    int indx = 0;
    int indx_q = 0;
    Value in_value;

    for (int i = 0; i < 2; ++i) {
      auto mul_op = dyn_cast<MulOp>(op.getInputs()[i].getDefiningOp());
      if (mul_op) {
        auto reshape_op = dyn_cast<ReshapeOp>(mul_op.getInputs()[0].getDefiningOp());
        if (reshape_op) {
          indx = i;
        }
        else{
          indx = 1-i;
        }
      }
      else{
        return failure();
      }
    }

    auto mul0_op = dyn_cast<MulOp>(op.getInputs()[indx].getDefiningOp());
    auto mul1_op = dyn_cast<MulOp>(op.getInputs()[1-indx].getDefiningOp());
     if (!mul0_op || !mul1_op) {
      return failure();
    }

    auto reshape_op =
        dyn_cast<ReshapeOp>(mul0_op.getInputs()[0].getDefiningOp());
    if (!reshape_op)
      return failure();
    auto mul0_weight = mul0_op.getInputs()[1];
    auto mul1_weight = mul1_op.getInputs()[1];
    auto concat_op = dyn_cast<ConcatOp>(reshape_op.getInput().getDefiningOp());
    if (!concat_op)
      return failure();
    for (int i = 0; i < 2; ++i) {
      auto unsqueeze = dyn_cast<UnsqueezeOp>(concat_op.getInputs()[i].getDefiningOp());
      if(unsqueeze){
        auto slice_op1 = dyn_cast<SliceOp>(unsqueeze.getInput().getDefiningOp());
        if (slice_op1){
          indx_q = i;
        }else{
          indx_q = 1 - i;
        }
      }
      else{
        return failure();
      }
    }

    auto unsqueeze0 =
        dyn_cast<UnsqueezeOp>(concat_op.getInputs()[indx_q].getDefiningOp());
    auto unsqueeze1 =
        dyn_cast<UnsqueezeOp>(concat_op.getInputs()[1-indx_q].getDefiningOp());
    if (!unsqueeze0)
      return failure();
    if (!unsqueeze1)
      return failure();

    auto slice_op1 = dyn_cast<SliceOp>(unsqueeze0.getInput().getDefiningOp());
    if (!slice_op1)
      return failure();
    auto MulConst_op =
        dyn_cast<MulConstOp>(unsqueeze1.getInput().getDefiningOp());
    if (!MulConst_op)
      return failure();
    auto slice_op = dyn_cast<SliceOp>(MulConst_op.getInput().getDefiningOp());
    if (!slice_op)
      return failure();
    if (mul1_op.getInputs()[0].getDefiningOp() ==
            slice_op.getInput().getDefiningOp() &&
        mul1_op.getInputs()[0].getDefiningOp() ==
            slice_op1.getInput().getDefiningOp() &&
        slice_op.getInput().getDefiningOp() ==
            slice_op1.getInput().getDefiningOp()) {
      in_value = slice_op.getInput();
              }
    else{
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }

    // Create RopeOp
    std::vector<NamedAttribute> attrs;
    rewriter.replaceOpWithNewOp<RopeOp>(op, op.getResult().getType(),
                                         ValueRange{in_value, mul0_weight, mul1_weight}, attrs);
    return success();
  }
};



struct AddToAddConst : public OpRewriterPatternEx<AddOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

    AddToAddConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<AddOp>(context, "AddToAddConst") {}

  LogicalResult matchAndRewriteImpl(AddOp op,
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
    if (!storage_type.isF32() && !storage_type.isF16())
      return failure();
    if (left_elt_num == 1) {
      if (auto left_op =
              dyn_cast_or_null<WeightOp>(op.getInputs()[0].getDefiningOp())) {
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
    if (!weight_flag) {
      return failure();
    }
    if (const_val->at(0) == 0.0f) {
      rewriter.replaceOp(op, {new_input});
      return success();
    }
    Type output = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val->at(0))));
    attrs.push_back(rewriter.getNamedAttr(
        "do_relu", op->getAttr("do_relu").cast<BoolAttr>()));
    attrs.push_back(rewriter.getNamedAttr(
        "relu_limit", op->getAttr("relu_limit").cast<FloatAttr>()));
    attrs.push_back(rewriter.getNamedAttr(
        "is_scalar", op->getAttr("is_scalar").cast<BoolAttr>()));
    rewriter.replaceOpWithNewOp<AddConstOp>(op, output, new_input, attrs);
    return success();
  }
};

// Add weight + Add Weight
struct AddMerge : public OpRewriterPatternEx<AddOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  AddMerge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<AddOp>(context, "AddMerge") {}

  LogicalResult matchAndRewriteImpl(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (module::isUniformQuantized(op.getOutput())) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16())
      return failure();
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
    auto b_data = b_op.read_as_float();
    if (b_coeff != 1.0) {
      for (auto &b : *b_data) {
        b *= b_coeff;
      }
    }
    auto d_data = d_op.read_as_float();
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
    rewriter.setInsertionPointAfter(op);
    auto weight = WeightOp::create_float(op, "merged", *output,
                                         o_shape, storage_type);
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

//[(5,16,1,32),(1,32)] -> [(5,16,1,32),(1,1,1,32)]
struct AlignInputsDim : public OpRewriterPatternEx<AddOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  AlignInputsDim(mlir::MLIRContext *context)
      : OpRewriterPatternEx<AddOp>(context, "AlignInputsDim") {}

  LogicalResult matchAndRewriteImpl(AddOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto lhs_shape = module::getShape(op.getInputs()[0]);
    auto rhs_shape = module::getShape(op.getInputs()[1]);
    if(lhs_shape.size() == rhs_shape.size())
      return failure();

    int diff_dims = lhs_shape.size() > rhs_shape.size() ? lhs_shape.size() - rhs_shape.size() : rhs_shape.size() - lhs_shape.size();
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> new_shape(diff_dims,1);
    Operation* add_need_reshape;
    if(lhs_shape.size() > rhs_shape.size()){
      add_need_reshape = op.getInputs()[1].getDefiningOp();
      for(auto shape : rhs_shape)
        new_shape.emplace_back(shape);
    } else {
      add_need_reshape = op.getInputs()[0].getDefiningOp();
      for(auto shape : lhs_shape)
        new_shape.emplace_back(shape);
    }
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape", rewriter.getI64ArrayAttr(
                    new_shape)));
    auto reshape_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(module::getName(op.getOutput()).str() + "_reshape")),
        RankedTensorType::get(new_shape, module::getElementType(add_need_reshape->getResult(0))),
        add_need_reshape->getResult(0), attrs);
    op.setOperand(lhs_shape.size() > rhs_shape.size() ? 1 : 0, reshape_op.getOutput());
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<SwapInput, AddToAddConst, AddToScale, AddToRope, AddMerge, AlignInputsDim>(context);
}
