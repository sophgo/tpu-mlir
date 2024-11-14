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
      if (where_op.getXIsConst()) {
        auto where_c_v = where_op.getXConstVal().convertToDouble();
        if (compare_c_v == where_c_v) {
          std::vector<NamedAttribute> attrs;
          attrs.push_back(rewriter.getNamedAttr(
              "const_val", rewriter.getF64FloatAttr(compare_c_v)));
          where_op.replaceAllUsesWith(op.getOutput());
          rewriter.replaceOpWithNewOp<MinConstOp>(op, op.getOutput().getType(),
                                                  op.getInput(), attrs);
        }
      }
    }
    return success();
  }
};

void CompareConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<CompareConstWhereToMinConst>(context);
}
