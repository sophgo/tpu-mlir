//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct CompareToCompareConst : public OpRewriterPatternEx<CompareOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  CompareToCompareConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<CompareOp>(context, "CompareToCompareConst") {}

  LogicalResult matchAndRewriteImpl(CompareOp op,
                                    PatternRewriter &rewriter) const override {

    auto right_shape = op.getRhs().getType().dyn_cast<TensorType>().getShape();
    int right_elt_num = 1;
    if (right_shape.size() > 1)
      return failure();
    for (int i = 0; i < right_shape.size(); ++i)
      right_elt_num *= right_shape[i];
    if (right_elt_num > 1)
      return failure();
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }

    std::shared_ptr<std::vector<float>> const_val;
    if (auto right_op =
            dyn_cast_or_null<WeightOp>(op.getRhs().getDefiningOp())) {
      const_val = right_op.read_as_float();
    } else {
      return failure();
    }

    Type output = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("mode", op.getModeAttr()));
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val->at(0))));
    attrs.push_back(rewriter.getNamedAttr("inversed", rewriter.getBoolAttr(0)));
    rewriter.replaceOpWithNewOp<CompareConstOp>(op, output, op.getLhs(), attrs);
    return success();
  }
};

/**
 * ConstantFill \
 *            Mul =>  Any -> MulConst(const=WeightData)
 * Any       /
 *
 */
struct CompareConstantFill : public OpRewriterPatternEx<CompareOp> {
public:
  CompareConstantFill(mlir::MLIRContext *context)
      : OpRewriterPatternEx<CompareOp>(context, "CompareConstantFill") {}

  LogicalResult matchAndRewriteImpl(CompareOp op,
                                    PatternRewriter &rewriter) const override {

    auto stype = module::getStorageType(op.getOutput());
    if (!stype.isF32()) {
      return failure();
    }
    auto op0 = op.getLhs().getDefiningOp();
    auto op1 = op.getRhs().getDefiningOp();
    Operation *const_op = nullptr;
    Operation *input_op = nullptr;
    bool inverse = false;
    Value new_input;
    if (isa_and_nonnull<top::ConstantFillOp>(op0)) {
      const_op = op0;
      input_op = op1;
      new_input = op.getRhs();
      inverse = true;
    } else if (isa_and_nonnull<top::ConstantFillOp>(op1)) {
      const_op = op1;
      input_op = op0;
      new_input = op.getLhs();
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
    Type otype = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("const_val", constOp.getValueAttr()));
    attrs.push_back(rewriter.getNamedAttr("mode", op.getModeAttr()));
    attrs.push_back(rewriter.getNamedAttr(
        "inversed", BoolAttr::get(getContext(), inverse)));
    rewriter.replaceOpWithNewOp<CompareConstOp>(op, otype, new_input, attrs);
    return success();
  }
};

void CompareOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<CompareToCompareConst, CompareConstantFill>(context);
}
