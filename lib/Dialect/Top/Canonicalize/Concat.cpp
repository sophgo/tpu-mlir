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

// A --slice--> A0 | A1 --concat--> A1 | A0
// ==> SwapDimInner
// test by `test_onnx.py --case SwapDimInner`
struct ConcatToSwapDimInner : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concat_op,
                                PatternRewriter &rewriter) const override {
    int num_inputs = concat_op.getInputs().size();
    if (num_inputs != 2) {
      return failure();
    }

    auto in0_op = concat_op.getInputs()[0].getDefiningOp();
    auto in1_op = concat_op.getInputs()[1].getDefiningOp();
    auto slice0_op = dyn_cast<SliceOp>(in0_op);
    auto slice1_op = dyn_cast<SliceOp>(in1_op);
    if (!slice0_op || !slice1_op) {
      return failure();
    }
    auto from = slice0_op.getInput();
    if (from != slice1_op.getInput()) {
      return failure();
    }
    auto steps0 = module::getI64Array(slice0_op.getSteps());
    auto steps1 = module::getI64Array(slice1_op.getSteps());
    for (int i = 0; i < steps0->size(); ++i) {
      if (steps0->at(i) != 1 || steps0->at(i) != steps1->at(i)) {
        return failure();
      }
    }
    auto offset0 = module::getI64Array(slice0_op.getOffset());
    auto offset1 = module::getI64Array(slice1_op.getOffset());
    // auto oshape0 = module::getShape(slice0_op.getOutput());
    auto oshape1 = module::getShape(slice1_op.getOutput());
    auto ishape = module::getShape(slice0_op.getInput());

    int axis = concat_op.getAxis();
    if (offset0->at(axis) != oshape1[axis] || offset1->at(axis) != 0) {
      return failure();
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(*offset0)));
    // concat_op->setLoc(NameLoc::get(rewriter.getStringAttr(
        // module::getName(concat_op.getOperation()).str() + "_SwapDimInner")));
    rewriter.replaceOpWithNewOp<SwapDimInnerOp>(
        concat_op, concat_op.getResult().getType(), ValueRange{from}, attrs);

    return success();
  }
};


static LogicalResult find_slice_order(ConcatOp concat_op, bool is_NCHW,
  std::vector<int64_t>& order, Value& from, int64_t& bh, int64_t& bw) {
  // idx of n,c,h,w
  int ni = 0, ci, hi, wi;
  if (is_NCHW) {
    ci = 1; hi = 2; wi = 3;
  } else {
    hi = 1; wi = 2; ci = 3;
  }
  bh = 0, bw = 0;
  const auto& inputs = concat_op.getInputs();
  int num_inputs = inputs.size();
  order.clear();
  order.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    auto in_op = inputs[i].getDefiningOp();
    if (!isa<SliceOp>(in_op)) {
      return failure();
    }
    auto slice_op = dyn_cast<SliceOp>(in_op);
    auto offset = module::getI64Array(slice_op.getOffset());
    if (offset->at(ni) != 0 || offset->at(ci) != 0) {
      return failure();
    }
    auto steps = module::getI64Array(slice_op.getSteps());
    if (steps->at(ni) != 1 || steps->at(ci) != 1) {
      return failure();
    }
    if (i == 0) {
      bh = steps->at(hi);
      bw = steps->at(wi);
      if (bh * bw != num_inputs) {
        return failure();
      }
      from = slice_op.getInput();
    } else {
      if (bh != steps->at(hi) || bw != steps->at(wi)) {
        return failure();
      }
      if (from != slice_op.getInput()) {
        return failure();
      }
    }
    int64_t begin_order = offset->at(hi) * bw + offset->at(wi);
    if (std::find(order.begin(), order.end(), begin_order) != order.end()) {
      return failure();
    }
    order.push_back(begin_order);
  }
  return success();
}

static void replaceOpWithDepth2SpaceOp(PatternRewriter& rewriter, ConcatOp& op,
  ValueRange&& args, int64_t bh, int64_t bw, bool is_CRD, bool is_inversed,
  bool in_is_NCHW, bool out_is_NCHW, bool swap_cr) {
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(bh)));
  attrs.push_back(
      rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(bw)));
  attrs.push_back(
      rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(is_CRD)));
  attrs.push_back(
      rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(is_inversed)));
  attrs.push_back(
      rewriter.getNamedAttr("in_is_NCHW", rewriter.getBoolAttr(in_is_NCHW)));
  attrs.push_back(
      rewriter.getNamedAttr("out_is_NCHW", rewriter.getBoolAttr(out_is_NCHW)));
  attrs.push_back(
      rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(swap_cr)));
  rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
      op, op.getResult().getType(), args, attrs);
}

// concat slices to Depth2Space.
// test by yolov5s
struct ConcatToDepth2SpacePattern : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concat_op,
                                PatternRewriter &rewriter) const override {
    auto shape = module::getShape(concat_op.getOutput());
    if (shape.size() != 4) {
      return failure();
    }
    if (concat_op.getAxis() != 1) {
      return failure();
    }
    if (concat_op->hasOneUse() == false) {
      return failure();
    }
    auto use_op = *concat_op->getUsers().begin();
    if (!isa<ConvOp>(use_op)) {
      return failure();
    }
    Value from; int64_t bh; int64_t bw;
    std::vector<int64_t> order;
    LogicalResult ret = find_slice_order(concat_op, true, order, from, bh, bw);
    if (ret.failed()) return failure();
    bool need_reorder = false;
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] != i && false == need_reorder) {
        need_reorder = true;
      }
    }
    if (need_reorder) {
      if (concat_op->hasOneUse() == false) {
        return failure();
      }
      auto use_op = *concat_op->getUsers().begin();
      if (!isa<ConvOp>(use_op)) {
        return failure();
      }
      auto conv_op = dyn_cast<ConvOp>(use_op);
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
    // change name of new op to avoid wrong comparison
    concat_op->setLoc(NameLoc::get(rewriter.getStringAttr(
        module::getName(concat_op.getOperation()).str() + "_Depth2Space")));
    replaceOpWithDepth2SpaceOp(rewriter, concat_op, ValueRange(from),
      bh, bw, false, true, true, true, false);
    return success();
  }
};

struct ConcatToDepth2SpacePattern2 : public OpRewritePattern<ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatOp concat_op,
                                PatternRewriter &rewriter) const override {
    const auto& shape = module::getShape(concat_op.getOutput());
    if (shape.size() != 4) {
      return failure();
    }
    if (concat_op.getAxis() != 1 && concat_op.getAxis() != 3) {
      return failure();
    }
    if (concat_op->hasOneUse()) {
      auto use_op = *concat_op->getUsers().begin();
      if (isa<ConvOp>(use_op)) {
        return failure();
      }
    }
    bool in_is_NCHW = concat_op.getAxis() == 1;
    bool out_is_NCHW = in_is_NCHW;
    Value from; int64_t bh; int64_t bw;
    std::vector<int64_t> order;
    LogicalResult ret = find_slice_order(concat_op, in_is_NCHW, order, from, bh, bw);
    if (ret.failed()) return failure();
    bool flag0 = true; bool flag1 = true;
    for (int64_t i = 0; i < bh * bw; ++i) {
      if (order[i] != i) {
        flag0 = false;
        break;
      }
    }
    if (!flag0) {
      for (int64_t i = 0; i < bw; ++i) {
        for (int64_t j = 0; j < bh; ++j) {
          if (order[j * bw + i] != i * bh + j) {
            flag1 = false;
            break;
          }
        }
      }
    }
    if (!flag0 && !flag1)
      return failure();
    bool swap_cr = flag1;
    replaceOpWithDepth2SpaceOp(rewriter, concat_op, ValueRange(from),
      bh, bw, false, true, in_is_NCHW, out_is_NCHW, swap_cr);
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
                 ConcatToDepth2SpacePattern, ConcatToDepth2SpacePattern2,
                 ConcatToSwapDimInner>(context);
}
