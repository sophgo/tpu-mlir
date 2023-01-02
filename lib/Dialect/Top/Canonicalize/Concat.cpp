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


using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

// concat slices to Depth2Space.
// test by yolov5s
struct ConcatToDepth2SpaceOp : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concat_op,
                                PatternRewriter &rewriter) const override {
    if (concat_op.getAxis() != 1) {
      return failure();
    }
    auto shape = module::getShape(concat_op.getOutput());
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
    int num_inputs = concat_op.getInputs().size();
    for (int i = 0; i < num_inputs; i++) {
      auto in_op = concat_op.getInputs()[i].getDefiningOp();
      auto slice_op = dyn_cast<SliceOp>(in_op);
      if (!slice_op) {
        return failure();
      }
      auto offset = module::getI64Array(slice_op.getOffset());
      if (offset->at(0) != 0 || offset->at(1) != 0) {
        return failure();
      }
      auto steps = module::getI64Array(slice_op.getSteps());
      if (steps->at(0) != 1 || steps->at(1) != 1) {
        return failure();
      }
      if (i == 0) {
        bh = steps->at(2);
        bw = steps->at(3);
        if (bh * bw != num_inputs) {
          return failure();
        }
        from = slice_op.getInput();
      } else {
        if (bh != steps->at(2) || bw != steps->at(3)) {
          return failure();
        }
        if (from != slice_op.getInput()) {
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
      if (conv_op.getGroup() != 1) {
        return failure();
      }
      auto filter_op = conv_op.getFilter().getDefiningOp<WeightOp>();
      // TODO: maybe filter is i8 in Top Dialect
      auto filter_old = filter_op.read<float>();
      auto filter_new =
          std::make_shared<std::vector<float>>(filter_old->size(), 0.0);
      int64_t oc, ic, kh, kw;
      module::getNCHW(conv_op.getFilter(), oc, ic, kh, kw);
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
      auto new_type = filter_op.getOutput().getType().cast<RankedTensorType>();
      auto new_filter_op =
          WeightOp::create(use_op, "filter_S2D", *filter_new, new_type);
      use_op->setOperand(1, new_filter_op);
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(bh)));
    attrs.push_back(
        rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(bw)));
    attrs.push_back(
        rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(false)));
    attrs.push_back(
        rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(true)));
    // change name of new op to avoid wrong comparison
    concat_op->setLoc(NameLoc::get(
        rewriter.getStringAttr(module::getName(concat_op.getOperation()).str() + "_Depth2Space")));
    rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
        concat_op, concat_op.getResult().getType(), ValueRange{from}, attrs);
    return success();
  }
};

struct ConvertLoadWeightConcatToLoadWeightPattern
    : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concat_op,
                                PatternRewriter &rewriter) const override {
    auto input_num = concat_op.getNumOperands();
    for (uint32_t i = 0; i < input_num; ++i) {
      auto formerOp = concat_op.getOperand(i).getDefiningOp();
      if (!isa<WeightOp>(formerOp)) {
        return failure();
      }
    }
    uint32_t h, w;
    int tmp_w = 0;

    auto o_shape = module::getShape(concat_op.getOutput());
    std::vector<float> resultT;

    std::vector<std::shared_ptr<std::vector<float>>> input_load_weight(
        input_num);

    for (uint32_t i = 0; i < input_num; ++i) {
      auto weight_op = cast<WeightOp>(concat_op.getOperand(i).getDefiningOp());
      input_load_weight[i] = weight_op.read<float>();
    }

    for (uint32_t i = 0; i < input_num; ++i) {
      auto w_shape = module::getShape(concat_op.getOperand(i));
      assert(3 == w_shape.size());
      h = w_shape[1];
      w = w_shape[2];

      float *input_data = (float *)input_load_weight[i]->data();
      for (uint32_t idx_h = 0; idx_h < h; ++idx_h) {
        std::vector<float> shapeT(w);
        int64_t insert_offset = ((idx_h + 1) * tmp_w) + idx_h * w;
        shapeT.assign(&input_data[idx_h * w], &input_data[(idx_h + 1) * w]);
        resultT.insert(resultT.begin() + insert_offset, shapeT.begin(),
                       shapeT.end());
      }
      tmp_w += w;
    }
    auto tensor_name = module::getName(concat_op, 0).str() + "loadweight";
    auto weight_type = RankedTensorType::get(o_shape, rewriter.getF32Type());
    auto weight_operand =
        WeightOp::create(concat_op, tensor_name, resultT, weight_type);
    rewriter.replaceOp(concat_op, weight_operand);
    return success();
  }
};

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ConvertLoadWeightConcatToLoadWeightPattern,
                 ConcatToDepth2SpaceOp>(context);
}
