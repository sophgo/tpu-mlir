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

// SLice after TopK may lead to some address error
// Slice erasing will begin under the 2 condition:
// 1. Values and Indices only have one SliceOp user, and the parameters of the
// two slices equal
// 2. Only one SliceOp user for both outputs
struct TopKWithSlice : public OpRewriterPatternEx<TopKOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopKWithSlice(mlir::MLIRContext *context)
      : OpRewriterPatternEx<TopKOp>(context, "TopKWithSlice") {}
  LogicalResult matchAndRewriteImpl(TopKOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isDynamic())
      return failure();
    if (op.getIndices().hasOneUse() && op.getValues().hasOneUse()) {
      return compare_slice(op, rewriter);
    } else if (op.getIndices().hasOneUse() && op.getValues().use_empty()) {
      if (auto slice_indices_op =
              dyn_cast_or_null<SliceOp>(*op.getIndices().getUsers().begin())) {
        return which_axes(op, slice_indices_op, NULL, rewriter);
      } else {
        return failure();
      }
    } else if (op.getValues().hasOneUse() && op.getIndices().use_empty()) {
      if (auto slice_values_op =
              dyn_cast_or_null<SliceOp>(*op.getValues().getUsers().begin())) {
        return which_axes(op, NULL, slice_values_op, rewriter);
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }

private:
  LogicalResult slice2k(TopKOp op, SliceOp slice_indice_op,
                        SliceOp slice_value_op, PatternRewriter &rewriter,
                        int64_t axis) const {
    SliceOp slice_op = slice_indice_op ? slice_indice_op : slice_value_op;
    auto slice_shape = module::getShape(slice_op);
    auto offset = module::getI64Array(slice_op.getOffset());
    auto steps = module::getI64Array(slice_op.getSteps());
    auto ends = module::getI64Array(slice_op.getEnds());

    auto topk_shape = module::getShape(slice_op);

    if (axis != op.getAxis()) {
      return failure();
    }
    if (1 != steps->at(axis)) {
      return failure();
    }
    // slice the end of the array
    if (offset->at(axis) != 0 && ends->at(axis) >= topk_shape[axis]) {
      auto topk_in_shape = module::getShape(op.getInput());
      // we can slice the end of the array, only when topk output all data
      if (topk_in_shape[axis] != op.getK()) {
        return failure();
      }
      bool largest = op.getLargest();
      op->setAttr("largest", rewriter.getBoolAttr(!largest));
      // If slice the middle of  the array, fail
    } else if (offset->at(axis) != 0 && ends->at(axis) < topk_shape[axis]) {
      return failure();
    }

    auto ndims = topk_shape.size();
    std::vector<int64_t> new_shape(ndims, 1);
    for (int64_t i = 0; i < ndims; i++) {
      new_shape[i] = topk_shape[i];
    }
    new_shape[axis] = slice_shape[axis];

    module::setShape(op.getIndices(), new_shape);
    module::setShape(op.getValues(), new_shape);

    int64_t K = slice_shape[axis];

    op->setAttr("K", rewriter.getI64IntegerAttr(K));

    std::vector<Location> locs = {};

    if (slice_indice_op) {
      std::string slice_indice_op_name =
          module::getName(slice_indice_op.getOperation()).str() + "_r_TopK";
      auto slice_loc =
          NameLoc::get(rewriter.getStringAttr(slice_indice_op_name));
      locs.push_back(slice_loc);
    } else {
      auto indices_loc = NameLoc::get(
          rewriter.getStringAttr(module::getName(op.getResult(0)).str()));
      locs.push_back(indices_loc);
    }
    if (slice_value_op) {
      std::string slice_value_op_name =
          module::getName(slice_value_op.getOperation()).str() + "_r_TopK";
      auto slice_loc =
          NameLoc::get(rewriter.getStringAttr(slice_value_op_name));
      locs.push_back(slice_loc);
    } else {
      auto values_loc = NameLoc::get(
          rewriter.getStringAttr(module::getName(op.getResult(1)).str()));
      locs.push_back(values_loc);
    }
    auto fused_loc = FusedLoc::get(getContext(), locs);
    op->setLoc(fused_loc);

    if (offset->at(axis) == 0 && ends->at(axis) < topk_shape[axis]) {
      return success();
      // reverse the order if slice the end of the array
    } else if (offset->at(axis) != 0 && ends->at(axis) >= topk_shape[axis]) {
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(axis)));

      auto slice_indices_op =
          dyn_cast_or_null<SliceOp>(*op.getIndices().getUsers().begin());
      auto slice_values_op =
          dyn_cast_or_null<SliceOp>(*op.getValues().getUsers().begin());
      rewriter.setInsertionPointAfter(op);
      if (slice_indices_op) {
        rewriter.replaceOpWithNewOp<ReverseOp>(
            slice_indices_op, slice_indices_op.getResult().getType(),
            slice_indices_op.getInput(), attrs);
      }
      if (slice_values_op) {
        rewriter.replaceOpWithNewOp<ReverseOp>(
            slice_values_op, slice_values_op.getResult().getType(),
            slice_values_op.getInput(), attrs);
      }
      return success();
    } else {
      return failure();
    }
  }

  LogicalResult which_axes(TopKOp op, SliceOp slice_indice_op,
                           SliceOp slice_value_op,
                           PatternRewriter &rewriter) const {
    SliceOp slice_op = slice_indice_op ? slice_indice_op : slice_value_op;
    if (!slice_op.getHasparamConvertAxesAttr().empty()) {
      auto axes = module::getI64Array(slice_op.getHasparamConvertAxesAttr());

      if (1 != axes->size()) {
        return failure();
      } else {
        auto axis = axes->at(0);
        return slice2k(op, slice_indice_op, slice_value_op, rewriter, axis);
      }
    } else {
      auto axes = module::getI64Array(slice_op.getAxes());
      if (1 != axes->size()) {
        return failure();
      } else {
        auto axis = axes->at(0);
        return slice2k(op, slice_indice_op, slice_value_op, rewriter, axis);
      }
    }
  }

  LogicalResult compare_slice(TopKOp op, PatternRewriter &rewriter) const {
    auto slice_indices_op =
        dyn_cast_or_null<SliceOp>(*op.getIndices().getUsers().begin());
    auto slice_values_op =
        dyn_cast_or_null<SliceOp>(*op.getValues().getUsers().begin());

    if (slice_indices_op || slice_values_op) {
      // two slice
      if (slice_indices_op && slice_values_op) {
        // two slices equal exactly
        if ((module::getShape(slice_indices_op) ==
             module::getShape(slice_values_op)) &
            (slice_indices_op.getHasparamConvertAxesAttr() ==
             slice_values_op.getHasparamConvertAxesAttr()) &
            (slice_indices_op.getAxes() == slice_values_op.getAxes()) &
            (slice_indices_op.getOffset() == slice_values_op.getOffset()) &
            (slice_indices_op.getSteps() == slice_values_op.getSteps()) &
            (slice_indices_op.getEnds() == slice_values_op.getEnds()) &
            (slice_indices_op.getOffsetT() == slice_values_op.getOffsetT()) &
            (slice_indices_op.getStepsT() == slice_values_op.getStepsT()) &
            (slice_indices_op.getEndsT() == slice_values_op.getEndsT())) {

          auto slice_ret =
              which_axes(op, slice_indices_op, slice_values_op, rewriter);
          return slice_ret;
        } else {
          return failure();
        }
        // once slice
      } else {
        if (slice_indices_op) {
          return which_axes(op, slice_indices_op, slice_values_op, rewriter);
        } else {
          return which_axes(op, slice_indices_op, slice_values_op, rewriter);
        }
      }
      // no slice
    } else {
      return failure();
    }
    return failure();
  }
};

struct TopKTranspose : public OpRewriterPatternEx<TopKOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopKTranspose(mlir::MLIRContext *context)
      : OpRewriterPatternEx<TopKOp>(context, "TopKTranspose") {}

  LogicalResult matchAndRewriteImpl(TopKOp op,
                                    PatternRewriter &rewriter) const override {
    // examine axis
    auto axis = op.getAxis();
    auto input = op.getInput();
    auto input_shape = module::getShape(input);
    int64_t rank = input_shape.size();

    // support axis < 0
    if (axis < 0) {
      axis += rank;
      op.setAxis(axis);
    }

    if (axis == rank - 1) {
      return failure();
    }

    // create permute
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != axis)
        perm.push_back(i);
    }
    perm.push_back(axis); // 将目标轴移到末尾

    // create invert permute
    std::vector<int64_t> inv_perm(rank);
    for (int64_t i = 0; i < rank; ++i) {
      inv_perm[perm[i]] = i;
    }

    // insert permuteOp
    rewriter.setInsertionPoint(op);
    auto perm_loc =
        NameLoc::get(rewriter.getStringAttr("transpose_before_topk"));
    auto perm_attr = rewriter.getI64ArrayAttr(perm);
    auto transpose_input =
        rewriter.create<PermuteOp>(perm_loc, input.getType(), input, perm_attr);

    // create new topk with axis == -1
    std::vector<Location> locs = {};
    std::string value_name = module::getName(op.getResult(0)).str();
    std::string indice_name = module::getName(op.getResult(1)).str();

    auto topk_value_loc =
        NameLoc::get(rewriter.getStringAttr(value_name + "_transpose"));
    auto topk_indice_loc =
        NameLoc::get(rewriter.getStringAttr(indice_name + "_transpose"));

    locs.push_back(topk_indice_loc);
    locs.push_back(topk_value_loc);
    auto fused_loc = FusedLoc::get(getContext(), locs);

    std::vector<Type> output_types{op.getValues().getType(),
                                   op.getIndices().getType()};
    auto new_topk = rewriter.create<TopKOp>(
        fused_loc, output_types, ValueRange{transpose_input.getOutput()},
        op->getAttrs());
    new_topk.setAxis(rank - 1); // axis = -1

    // permute values
    auto value_loc = NameLoc::get(rewriter.getStringAttr(value_name));
    auto transpose_values = rewriter.create<PermuteOp>(
        value_loc, op.getValues().getType(), new_topk.getValues(),
        rewriter.getI64ArrayAttr(inv_perm));

    // permute indices
    auto indice_loc = NameLoc::get(rewriter.getStringAttr(indice_name));
    auto transpose_indices = rewriter.create<PermuteOp>(
        indice_loc, op.getIndices().getType(), new_topk.getIndices(),
        rewriter.getI64ArrayAttr(inv_perm));

    rewriter.replaceOp(
        op, {transpose_values.getOutput(), transpose_indices.getOutput()});

    return success();
  }
};

void TopKOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<TopKWithSlice>(context);
  results.insert<TopKTranspose>(context);
}
