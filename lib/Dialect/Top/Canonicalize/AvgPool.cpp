//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir::top;
using namespace tpu_mlir::helper;

struct AvgPoolToDwConv : public OpRewritePattern<AvgPoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AvgPoolOp op,
                                PatternRewriter &rewriter) const override {
    int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
    bool relu, is_global, count_include_pad;
    op.parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr,
                  pad_value, relu, is_global, count_include_pad);
    if (count_include_pad == false) {
      return failure();
    }
    std::vector<int64_t> filter_shape = {c, 1, kh, kw};
    std::vector<float> filter(c * kh * kw, 1.0 / (kh * kw));
    auto filter_type =
        RankedTensorType::get(filter_shape, rewriter.getF32Type());
    auto new_filter = WeightOp::create(op, "avarage", filter, filter_type);
    std::vector<Value> operands;
    operands.push_back(op.input());
    operands.push_back(new_filter);
    auto none = Module::getNoneOp(op.getOperation());
    operands.push_back(none);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
    attrs.push_back(rewriter.getNamedAttr("kernel_shape",
                                          rewriter.getI64ArrayAttr({kh, kw})));
    attrs.push_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({sh, sw})));
    attrs.push_back(rewriter.getNamedAttr(
        "pads", rewriter.getI64ArrayAttr({pt, pl, pb, pr})));
    attrs.push_back(
        rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(c)));
    attrs.push_back(
        rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(relu)));
    auto newOp = rewriter.create<ConvOp>(op.getLoc(), op.output().getType(),
                                         ArrayRef<Value>{operands},
                                         ArrayRef<NamedAttribute>{attrs});
    rewriter.replaceOp(op, {newOp.getResult()});
    return success();
  }
};

void AvgPoolOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
//  results.insert<AvgPoolToDwConv>(context);
}
