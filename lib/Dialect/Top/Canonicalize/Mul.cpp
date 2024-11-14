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

struct MulToSiLU : public OpRewriterPatternEx<MulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MulToSiLU(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MulOp>(context, "MulToSiLU") {}

  LogicalResult matchAndRewriteImpl(MulOp op,
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

/**
 * Weight[1] \
 *            Mul =>  Any -> MulConst(const=WeightData)
 * Any       /
 *
 */
struct MulToMulConst : public OpRewriterPatternEx<MulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MulToMulConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MulOp>(context, "MulToMulConst") {}

  LogicalResult matchAndRewriteImpl(MulOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32()) {
      return failure();
    }

    int is_const[2];

    is_const[0] = module::getNumElements(op.getInputs()[0]) == 1;
    is_const[1] = module::getNumElements(op.getInputs()[1]) == 1;
    if (!is_const[0] && !is_const[1]) {
      return failure();
    }

    Value new_input;
    std::shared_ptr<std::vector<float>> const_val;
    int weight_index = -1;

    for (int i = 0; i < 2; i++) {
      if (!is_const[i]) {
        continue;
      }
      if (auto weight_op =
              dyn_cast<WeightOp>(op.getInputs()[i].getDefiningOp())) {
        const_val = weight_op.read_as_float();
        weight_index = i;
        new_input = op.getInputs()[1 - i]; // take another operand as new input
        break;
      }
    }

    if (weight_index == -1) {
      return failure();
    }

    if (std::fabs(const_val->at(0) - 1.0f) < 2e-5f) {
      // erase mul
      rewriter.replaceOp(op, {new_input});
      return success();
    }
    Type output = new_input.getType();
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

/**
 * ConstantFill \
 *            Mul =>  Any -> MulConst(const=WeightData)
 * Any       /
 *
 */
struct MulToMulConst2 : public OpRewriterPatternEx<MulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MulToMulConst2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MulOp>(context, "MulToMulConst2") {}

  LogicalResult matchAndRewriteImpl(MulOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto stype = module::getStorageType(op.getOutput());
    if (!stype.isF32()) {
      return failure();
    }
    auto op0 = op.getInputs()[0].getDefiningOp();
    auto op1 = op.getInputs()[1].getDefiningOp();
    Operation *const_op = nullptr;
    Operation *input_op = nullptr;
    Value new_input;
    if (isa_and_nonnull<top::ConstantFillOp>(op0)) {
      const_op = op0;
      input_op = op1;
      new_input = op.getInputs()[1];
    } else if (isa_and_nonnull<top::ConstantFillOp>(op1)) {
      const_op = op1;
      input_op = op0;
      new_input = op.getInputs()[0];
    } else {
      return failure();
    }
    auto constOp = cast<top::ConstantFillOp>(const_op);
    auto in_shape = module::getShape(new_input);
    auto c_shape = module::getShape(constOp.getOutput());
    if (module::getNumElements(constOp.getOutput()) == 1) {
    } else if (in_shape.size() == c_shape.size()) {
      for (auto it : llvm::zip(in_shape, c_shape)) {
        if (std::get<0>(it) < std::get<1>(it)) {
          // shape broadcast
          return failure();
        }
      }
    } else {
      return failure();
    }
    auto const_val = constOp.getValue().convertToDouble();
    if (const_val == 1.0) {
      rewriter.replaceOp(op, {new_input});
      return success();
    }
    Type otype = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("const_val",
                                          rewriter.getF64FloatAttr(const_val)));
    attrs.push_back(rewriter.getNamedAttr(
        "do_relu", op->getAttr("do_relu").cast<BoolAttr>()));
    attrs.push_back(rewriter.getNamedAttr(
        "relu_limit", op->getAttr("relu_limit").cast<FloatAttr>()));
    rewriter.replaceOpWithNewOp<MulConstOp>(op, otype, new_input, attrs);
    return success();
  }
};

struct MulToScale : public OpRewriterPatternEx<MulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MulToScale(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MulOp>(context, "MulToScale") {}

  LogicalResult matchAndRewriteImpl(MulOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (module::isUniformQuantized(op.getInputs()[0]) == true) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
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

    // std::vector<float> scale(left_shape[1]);
    if (!dyn_cast<WeightOp>(S.getDefiningOp()))
      return failure();

    std::vector<float> bias(left_shape[1], 0);
    // module::gets
    auto B = WeightOp::create_float(op, "bias", bias, module::getShape(S).vec(),
                                    storage_type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "do_relu", op->getAttr("do_relu").cast<BoolAttr>()));
    attrs.push_back(rewriter.getNamedAttr(
        "relu_limit", op->getAttr("relu_limit").cast<FloatAttr>()));

    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getOutput().getType(),
                                         ValueRange{X, S, B}, attrs);
    return success();
  }
};

// Mul + Mul
struct MulMerge : public OpRewriterPatternEx<MulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MulMerge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MulOp>(context, "MulMerge") {}

  LogicalResult matchAndRewriteImpl(MulOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16())
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
    auto b_data = b_op.read_as_float();
    auto d_data = d_op.read_as_float();
    auto b_shape = module::getShape(b);
    auto d_shape = module::getShape(d);
    std::vector<int64_t> o_shape;
    auto output =
        binary_mul(b_data->data(), d_data->data(), b_shape, d_shape, o_shape);
    rewriter.setInsertionPointAfter(op);
    auto weight =
        WeightOp::create_float(op, "divisor", *output, o_shape, storage_type);
    std::vector<NamedAttribute> attrs;
    if (op.getDoRelu()) {
      attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
      attrs.push_back(
          rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    }
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), ValueRange{c, weight},
                                       attrs);
    if (mul->getUsers().empty()) {
      rewriter.eraseOp(mul);
    }
    return success();
  }
};

struct MergeGelu : public OpRewriterPatternEx<MulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MergeGelu(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MulOp>(context, "MergeGelu") {}

  LogicalResult matchAndRewriteImpl(MulOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isUniformQuantized(op.getOutput()))
      return failure();
    if (!op.getResult().hasOneUse())
      return failure();
    MulConstOp mulconst_op = NULL;
    AddConstOp addconst_op = NULL;
    for (auto in : op.getInputs()) {
      if (auto weight_op = dyn_cast<WeightOp>(in.getDefiningOp()))
        return failure();
      else if ((addconst_op = dyn_cast<AddConstOp>(in.getDefiningOp())))
        continue;
      else if ((mulconst_op = dyn_cast<MulConstOp>(in.getDefiningOp())))
        continue;
      else
        return failure();
    }
    if (mulconst_op == NULL || addconst_op == NULL)
      return failure();
    if (!mulconst_op.getResult().hasOneUse() ||
        !addconst_op.getResult().hasOneUse())
      return failure();
    if (fabs(mulconst_op.getConstVal().convertToDouble() - 0.5) > 1e-4)
      return failure();
    if (fabs(addconst_op.getConstVal().convertToDouble() - 1.0) > 1e-4)
      return failure();
    ErfOp erf_op = NULL;
    erf_op = dyn_cast<ErfOp>(addconst_op.getInput().getDefiningOp());
    if (erf_op == NULL)
      return failure();
    if (!erf_op.getResult().hasOneUse())
      return failure();
    MulConstOp mulconst_op1 = NULL;
    mulconst_op1 = dyn_cast<MulConstOp>(erf_op.getInput().getDefiningOp());
    if (mulconst_op1 == NULL)
      return failure();
    if (fabs(mulconst_op1.getConstVal().convertToDouble() -
             0.70710676908493042f) > 1e-4)
      return failure();
    if (mulconst_op1.getInput().getDefiningOp() !=
        mulconst_op.getInput().getDefiningOp())
      return failure();
    rewriter.replaceOpWithNewOp<GELUOp>(op, op.getResult().getType(),
                                        ValueRange{mulconst_op.getInput()});
    return success();
  }
};

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MulToSiLU, MulToMulConst, MulToMulConst2, MulToScale, MulMerge,
                 MergeGelu>(context);
}
