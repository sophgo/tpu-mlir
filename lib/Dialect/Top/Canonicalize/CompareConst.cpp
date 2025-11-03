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

struct CompareConstWhereToMinConst
    : public OpRewriterPatternEx<CompareConstOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  CompareConstWhereToMinConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<CompareConstOp>(context,
                                            "CompareConstWhereToMinConst") {}

  LogicalResult matchAndRewriteImpl(CompareConstOp op,
                                    PatternRewriter &rewriter) const override {

    auto compare_c_v = op.getConstVal().convertToDouble();
    if (op.getMode().str() == "Less" && op.getResult().hasOneUse() &&
        isa<WhereOp>(*op.getResult().getUsers().begin())) {
      auto where_op = dyn_cast<WhereOp>(*op.getResult().getUsers().begin());
      auto where_loc = module::getLoc(where_op.getOutput());
      if (where_op.getXIsConst()) {
        auto where_c_v = where_op.getXConstVal().convertToDouble();
        if (compare_c_v == where_c_v) {
          std::vector<NamedAttribute> attrs;
          attrs.push_back(rewriter.getNamedAttr(
              "const_val", rewriter.getF64FloatAttr(compare_c_v)));
          where_op.replaceAllUsesWith(op.getOutput());
          auto new_ctx = rewriter.replaceOpWithNewOp<MinConstOp>(
              op, op.getOutput().getType(), op.getInput(), attrs);
          module::setLoc(new_ctx, where_loc);
        }
      }
    }
    return success();
  }
};

struct FuseNotEqualPattern : public OpRewriterPatternEx<CompareConstOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  FuseNotEqualPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<CompareConstOp>(context, "FuseNotEqualPattern") {}

  LogicalResult matchAndRewriteImpl(CompareConstOp eqOp,
                                    PatternRewriter &rewriter) const override {
    // Not(Equal(x, c))  ==> CompareConst(mode="NotEqual", const_val=c)
    if (eqOp.getMode().str() != "Equal")
      return failure();

    Value eqOut = eqOp.getOutput(); // or eqOp.getResult()
    if (!eqOut.hasOneUse())
      return failure();

    Operation *user = *eqOut.getUsers().begin();
    auto notOp = dyn_cast<CompareConstOp>(user);
    if (!notOp)
      return failure();
    if (notOp.getMode().str() != "Not")
      return failure();

    double constVal = eqOp.getConstVal().convertToDouble();
    bool isScalar = eqOp.getIsScalar();
    bool inversed = false;

    Value dataInput = eqOp.getInput();

    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(constVal)));
    attrs.push_back(
        rewriter.getNamedAttr("inversed", rewriter.getBoolAttr(inversed)));
    attrs.push_back(
        rewriter.getNamedAttr("is_scalar", rewriter.getBoolAttr(isScalar)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("NotEqual")));

    auto outType = notOp.getOutput().getType();

    // CompareConst(mode="NotEqual")
    rewriter.replaceOpWithNewOp<CompareConstOp>(notOp, outType, dataInput,
                                                attrs);

    if (eqOp.use_empty()) {
      rewriter.eraseOp(eqOp);
    }

    return success();
  }
};

void CompareConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<FuseNotEqualPattern, CompareConstWhereToMinConst>(context);
}
