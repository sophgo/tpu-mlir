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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

// concat slices to Depth2Space.
// test by yolov5s
struct ConcatToDepth2SpaceOp : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concat_op,
                                PatternRewriter &rewriter) const override {
    if (concat_op.axis() != 1) {
      return failure();
    }
    auto shape = Module::getShape(concat_op.output());
    if (shape.size() != 4) {
      return failure();
    }
    if (concat_op->hasOneUse() == false) {
      return failure();
    }
    auto use_op = *concat_op->getUsers().begin();
    auto conv_op = dyn_cast<ConvOp>(use_op);
    if (!conv_op) {
      return failure();
    }
    int64_t bh = 0, bw = 0;
    Value from;
    std::vector<int64_t> order;
    bool need_reorder = false;
    int num_inputs = concat_op.inputs().size();
    for (int i = 0; i < num_inputs; i++) {
      auto in_op = concat_op.inputs()[i].getDefiningOp();
      auto slice_op = dyn_cast<SliceOp>(in_op);
      if (!slice_op) {
        return failure();
      }
      auto offset = Module::getI64Array(slice_op.offset());
      if (offset->at(0) != 0 || offset->at(1) != 0) {
        return failure();
      }
      auto steps = Module::getI64Array(slice_op.steps());
      if (steps->at(0) != 1 || steps->at(1) != 1) {
        return failure();
      }
      if (i == 0) {
        bh = steps->at(2);
        bw = steps->at(3);
        if (bh * bw != num_inputs) {
          return failure();
        }
        from = slice_op.input();
      } else {
        if (bh != steps->at(2) || bw != steps->at(3)) {
          return failure();
        }
        if (from != slice_op.input()) {
          return failure();
        }
      }
      int64_t begin_order = offset->at(2) * bw + offset->at(3);
      if (std::find(order.begin(), order.end(), begin_order) != order.end()) {
        return failure();
      }
      order.push_back(begin_order);
      if (begin_order != i && false == need_reorder) {
        need_reorder = true;
      }
    }
    if (need_reorder) {
      if (concat_op->hasOneUse() == false) {
        return failure();
      }
      auto use_op = *concat_op->getUsers().begin();
      auto conv_op = dyn_cast<ConvOp>(use_op);
      if (!conv_op) {
        return failure();
      }
      if (conv_op.group() != 1) {
        return failure();
      }
      auto filter_op = conv_op.filter().getDefiningOp<WeightOp>();
      // TODO: maybe filter is i8 in Top Dialect
      auto filter_old = filter_op.read<float>();
      auto filter_new =
          std::make_shared<std::vector<float>>(filter_old->size(), 0.0);
      int64_t oc, ic, kh, kw;
      Module::getNCHW(conv_op.filter(), oc, ic, kh, kw);
      int64_t block = bh * bw;
      int64_t inner_dim = (ic / block) * kh * kw;
      int64_t outer_dim = oc;
      for (int o = 0; o < outer_dim; o++) {
        for (int i = 0; i < block; i++) {
          auto begin = filter_old->begin() + (o * block + order[i]) * inner_dim;
          auto end = begin + inner_dim;
          auto to = filter_new->begin() + (o * block + i) * inner_dim;
          std::copy(begin, end, to);
        }
      }
      auto new_type = filter_op.output().getType().cast<RankedTensorType>();
      auto new_filter_op =
          WeightOp::create(use_op, "filter_S2D", *filter_new, new_type);
      use_op->setOperand(1, new_filter_op);
    }
    std::vector<NamedAttribute> attrs;
    if (need_reorder == false) {
      attrs.push_back(rewriter.getNamedAttr("name", concat_op.nameAttr()));
    } else {
      attrs.push_back(rewriter.getNamedAttr(
          "name", rewriter.getStringAttr(concat_op.name() + "_S2D")));
    }
    attrs.push_back(
        rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(bh)));
    attrs.push_back(
        rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(bw)));
    attrs.push_back(
        rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(false)));
    attrs.push_back(
        rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(true)));
    rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
        concat_op, concat_op.getResult().getType(), ValueRange{from}, attrs);
    return success();
  }
};

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ConcatToDepth2SpaceOp>(context);
}
