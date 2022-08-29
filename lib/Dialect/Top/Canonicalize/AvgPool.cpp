//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"

using namespace mlir;
using namespace tpu_mlir::top;
using namespace tpu_mlir::helper;

struct AvgPoolToDwConv : public OpRewritePattern<AvgPoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AvgPoolOp op,
                                PatternRewriter &rewriter) const override {
    tpu_mlir::pool_attr_t param;
    op.parseParam(&param);
    if (param.count_include_pad == false || op.kernel_shape().size() == 3) {
      return failure();
    }
    std::vector<int64_t> filter_shape = {param.c, 1, param.kh, param.kw};
    std::vector<float> filter(param.c * param.kh * param.kw,
                              1.0 / (param.kh * param.kw));
    auto filter_type =
        RankedTensorType::get(filter_shape, rewriter.getF32Type());
    auto new_filter = WeightOp::create(op, "avarage", filter, filter_type);
    std::vector<Value> operands;
    operands.push_back(op.input());
    operands.push_back(new_filter);
    auto none = Module::getNoneOp(op.getOperation());
    operands.push_back(none);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "kernel_shape", rewriter.getI64ArrayAttr({param.kh, param.kw})));
    attrs.push_back(rewriter.getNamedAttr(
        "strides", rewriter.getI64ArrayAttr({param.sh, param.sw})));
    attrs.push_back(rewriter.getNamedAttr(
        "pads",
        rewriter.getI64ArrayAttr(
            {param.pad_h, param.pad_w, param.pad_h_after, param.pad_w_after})));
    attrs.push_back(
        rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(param.c)));
    attrs.push_back(
        rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(param.do_relu)));
    auto newOp = rewriter.create<ConvOp>(op.getLoc(), op.output().getType(),
                                         operands, attrs);
    rewriter.replaceOp(op, {newOp.getResult()});
    return success();
  }
};

void AvgPoolOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  //  results.insert<AvgPoolToDwConv>(context);
}
