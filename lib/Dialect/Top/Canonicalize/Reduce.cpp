//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include <algorithm>

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct TopReduceTranspose : public OpRewriterPatternEx<ReduceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopReduceTranspose(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewriterPatternEx<ReduceOp>(context, "TopReduceTranspose", benefit) {}

  LogicalResult matchAndRewriteImpl(ReduceOp op,
                                    PatternRewriter &rewriter) const override {

    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<PermuteOp>(formerOp)) {
      return failure();
    }
    auto permuteOp = cast<PermuteOp>(formerOp);
    if (op.getKeepdims()) {
      return failure();
    }
    auto reduce_axes = module::getI64Array(op.getAxes());
    auto permute_order = module::getI64Array(permuteOp.getOrder());
    auto input_shape = module::getShape(op.getInput());
    auto input_dim = input_shape.size();
    // reduce all axes, permute can be erased if not used by others
    if (input_dim == reduce_axes->size()) {
      op->setOperand(0, permuteOp.getInput());
      if (permuteOp.getOutput().getUsers().empty()) {
        rewriter.eraseOp(permuteOp);
      }
      return success();
    }
    std::vector<int64_t> new_axis(reduce_axes->size(), 0);
    for (int i = 0; i < reduce_axes->size(); i++) {
      int axis = reduce_axes->at(i);
      axis = axis < 0 ? (axis + input_dim) : axis;
      assert((axis >= 0 && axis < input_dim) && "0 <= axis < in_dims");
      new_axis[i] = permute_order->at(axis);
    }
    /*
    fuse need satisfied two conditions belowL
    1. for input shape all the transposed axes is closed to each other, in other
    word, we have only one "continuous transposed axes range"
    2. in the transposed range, we have less than one dim after reduce
    */
    int left_count = 0;
    int continuous_transposed_range_num = 0;
    int continuous_transposed_sign = 0;
    for (int i = 0; i < input_dim; i++) {
      if (permute_order->at(i) != i) {
        left_count++;
        if (0 == continuous_transposed_sign) {
          continuous_transposed_range_num++;
          continuous_transposed_sign = 1;
        }
        for (int j = 0; j < reduce_axes->size(); j++) {
          if (new_axis[j] == permute_order->at(i)) {
            left_count--;
            break;
          }
        }
      } else if (1 == continuous_transposed_sign) {
        continuous_transposed_sign = 0;
      }
    }
    if (left_count > 1 || continuous_transposed_range_num > 1) {
      return failure();
    }
    std::sort(new_axis.begin(), new_axis.end());
    op->setAttr("axes", rewriter.getI64ArrayAttr(new_axis));
    op->setOperand(0, permuteOp.getInput());
    if (permuteOp.getOutput().getUsers().empty()) {
      rewriter.eraseOp(permuteOp);
    }
    return success();
  }
};

struct ReduceFusePattern : public OpRewriterPatternEx<ReduceOp> {

  using OpRewriterPatternEx::OpRewriterPatternEx;
  ReduceFusePattern(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewriterPatternEx<ReduceOp>(context, " ReduceFusePattern", benefit) {}
  LogicalResult matchAndRewriteImpl(ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    auto formerOpDefine = op.getInput().getDefiningOp();
    if (!isa<ReduceOp>(op) || !isa<ReduceOp>(formerOpDefine)) {
      return failure();
    }
    auto formerOp = cast<ReduceOp>(formerOpDefine);
    if (formerOp.getMode() != op.getMode()) {
      return failure();
    }
    if (formerOp.getKeepdims() != op.getKeepdims()) {
      return failure();
    }
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }
    auto formerOpShape = module::getShape(formerOp.getInput());
    auto formerOpDims = formerOpShape.size();
    if (formerOpShape[formerOpDims - 1] * formerOpShape[formerOpDims - 2] >=
        65536) {
      return failure();
    }
    auto axis_list_former = module::getI64Array(formerOp.getAxes());
    auto axis_list_current = module::getI64Array(op.getAxes());
    auto new_input = formerOp.getInput();

    int64_t N = formerOpShape.size();
    std::vector<int64_t> new_axis;
    new_axis.reserve(axis_list_former->size() + axis_list_current->size());

    if (!op.getKeepdims()) {
      std::vector<bool> removed(N, false);
      for (auto a : *axis_list_former) {
        removed[a] = true;
      }
      std::vector<int64_t> kept;
      kept.reserve(N - (int64_t)axis_list_former->size());
      for (int64_t i = 0; i < N; ++i) {
        if (!removed[i])
          kept.push_back(i);
      }

      std::vector<int64_t> current_axes_mapped;
      current_axes_mapped.reserve(axis_list_current->size());
      for (auto a : *axis_list_current) {
        current_axes_mapped.push_back(kept[a]);
      }

      llvm::SmallDenseSet<int64_t> set_axes;
      for (auto a : *axis_list_former)
        set_axes.insert(a);
      for (auto a : current_axes_mapped)
        set_axes.insert(a);
      new_axis.assign(set_axes.begin(), set_axes.end());
      std::sort(new_axis.begin(), new_axis.end());
    } else {
      llvm::SmallDenseSet<int64_t> set_axes;
      for (auto a : *axis_list_former) {
        set_axes.insert(a);
      }
      for (auto a : *axis_list_current) {
        set_axes.insert(a);
      }
      new_axis.assign(set_axes.begin(), set_axes.end());
      std::sort(new_axis.begin(), new_axis.end());
    }
    // check new_axis is continuous, backend not support cross-axes reduce
    for (int i = 1; i < new_axis.size(); i++) {
      if (new_axis[i] != new_axis[i - 1] + 1) {
        return failure();
      }
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(new_axis)));
    attrs.push_back(rewriter.getNamedAttr("keepdims", op.getKeepdimsAttr()));
    attrs.push_back(rewriter.getNamedAttr("mode", op.getModeAttr()));
    rewriter.replaceOpWithNewOp<ReduceOp>(op, op.getResult().getType(),
                                          new_input, attrs);
    return success();
  }
};

struct ReduceToReshapePattern : public OpRewriterPatternEx<ReduceOp> {

  using OpRewriterPatternEx::OpRewriterPatternEx;
  ReduceToReshapePattern(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewriterPatternEx<ReduceOp>(context, "ReduceToReshapePattern",
                                      benefit) {}
  LogicalResult matchAndRewriteImpl(ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    auto nofInputElts = module::getNumElements(op.getInput());
    auto nofOutputElts = module::getNumElements(op.getOutput());
    if (nofInputElts != nofOutputElts) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getInput(),
                                           std::vector<NamedAttribute>());
    return success();
  }
};

struct ReduceToRPoolPattern : public OpRewriterPatternEx<ReduceOp> {

  using OpRewriterPatternEx::OpRewriterPatternEx;
  ReduceToRPoolPattern(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewriterPatternEx<ReduceOp>(context, "ReduceToRPoolPattern",
                                      benefit) {}
  LogicalResult matchAndRewriteImpl(ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    auto iShape = module::getShape(op.getInput());
    auto num_dims = iShape.size();
    auto mode = op.getMode().str();
    auto axes = module::getI64Array(op.getAxes());
    if ((mode == "ReduceMean" || mode == "ReduceMax") && num_dims == 4 &&
        axes->size() == 2 && axes->at(0) == num_dims - 2 &&
        axes->at(1) == num_dims - 1) {
      // prepare attrs
      std::vector<NamedAttribute> pool_attrs;
      pool_attrs.emplace_back(rewriter.getNamedAttr(
          "kernel_shape", rewriter.getI64ArrayAttr(
                              {iShape[num_dims - 2], iShape[num_dims - 1]})));
      pool_attrs.emplace_back(
          rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
      pool_attrs.emplace_back(rewriter.getNamedAttr(
          "pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
      pool_attrs.emplace_back(rewriter.getNamedAttr(
          "count_include_pad", rewriter.getBoolAttr(true)));
      pool_attrs.emplace_back(
          rewriter.getNamedAttr("keepdims", op.getKeepdimsAttr()));
      if (mode == "ReduceMean") {
        rewriter.replaceOpWithNewOp<AvgPoolOp>(op, op.getType(), op.getInput(),
                                               pool_attrs);
      } else {
        rewriter.replaceOpWithNewOp<MaxPoolOp>(op, op.getType(), op.getInput(),
                                               pool_attrs);
      }
      return success();
    } else {
      return failure();
    }
  }
};

struct ReduceDiscontinuousPattern : public OpRewriterPatternEx<ReduceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  ReduceDiscontinuousPattern(MLIRContext *context, PatternBenefit benefit = 7)
      : OpRewriterPatternEx<ReduceOp>(context, "ReduceDiscontinuousPattern",
                                      benefit) {}
  LogicalResult matchAndRewriteImpl(ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    auto axes_val = module::getI64Array(op.getAxes());
    auto ishape = module::getShape(op.getInput());
    auto max_idx = *std::max_element(axes_val->begin(), axes_val->end());
    auto min_idx = *std::min_element(axes_val->begin(), axes_val->end());
    std::vector<int64_t> axes;
    for (int i = min_idx; i <= max_idx; i++) {
      axes.push_back(i);

      if (!std::count(axes_val->begin(), axes_val->end(), i) &&
          ishape[i] != 1) {
        return failure();
      }
    }
    if (axes_val->size() == axes.size()) {
      return failure();
    }
    std::sort(axes.begin(), axes.end());
    if (op.getKeepdims()) {
      op->setAttr("axes", rewriter.getI64ArrayAttr(axes));
    } else {
      std::string name = module::getName(op.getResult()).str();
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_r_0"));
      auto oshape = ishape.vec();
      for (int j = axes.size() - 1; j >= 0; j--) {
        oshape.erase(oshape.begin() + axes[j]);
      }
      auto type =
          RankedTensorType::get(oshape, module::getElementType(op.getOutput()));
      auto reduce_op = rewriter.create<top::ReduceOp>(
          loc, type, ValueRange{op.getInput()}, op->getAttrs());
      reduce_op->setAttr("axes", rewriter.getI64ArrayAttr(axes));
      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(),
                                             reduce_op.getResult(),
                                             std::vector<NamedAttribute>());
    }
    return success();
  }
};

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results
      .insert<ReduceToReshapePattern, ReduceToRPoolPattern, TopReduceTranspose,
              ReduceFusePattern, ReduceDiscontinuousPattern>(context);
}
