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
#include <vector>

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

struct SplitSlicePattern : public OpRewriterPatternEx<SliceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  SplitSlicePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceOp>(context, "SplitSlicePattern") {}

  LogicalResult matchAndRewriteImpl(SliceOp op,
                                    PatternRewriter &rewriter) const override {
    if (op->hasOneUse()) {
      return failure();
    }
    if (module::isDynamic()) {
      // to be consistent with MergeSlicePattern
      return failure();
    }
    // ****** CAUTION ******
    // Method 1:
    // auto users = op->getUsers();
    // for (auto user: users) {
    //  *** remove users ***
    // }
    // Method 2:
    // for (auto user: op->getUsers()) {
    //  *** remove users ***
    // }
    // Both methods might cause problems! We should cache the users first!
    std::vector<mlir::Operation *> users;
    for (auto user : op->getUsers()) {
      if (!isa<SliceOp>(user)) {
        return failure();
      }
      if (!module::isNone(dyn_cast<SliceOp>(*user).getOffsetT()) ||
          !module::isNone(dyn_cast<SliceOp>(*user).getEndsT()) ||
          !module::isNone(dyn_cast<SliceOp>(*user).getStepsT())) {
        return failure();
      }
      users.emplace_back(user);
    }
    const auto &opd = op->getOperand(0);
    const auto &res = op->getResult(0);
    const auto &offset = op->getAttr("offset");
    const auto &steps = op->getAttr("steps");
    const auto &ends = op->getAttr("ends");
    const auto &axes = op->getAttr("hasparamConvert_axes");
    rewriter.setInsertionPointAfterValue(opd);
    for (const auto user : users) {
      const std::string name_slice =
          module::getName(user->getOperand(0)).str() + "_r_slice";
      const auto &loc_slice = NameLoc::get(rewriter.getStringAttr(name_slice));
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("offset", offset));
      attrs.push_back(rewriter.getNamedAttr("steps", steps));
      attrs.push_back(rewriter.getNamedAttr("ends", ends));
      attrs.push_back(rewriter.getNamedAttr("hasparamConvert_axes", axes));
      auto none = module::getNoneOp(op);
      std::vector<Value> operands;
      operands.push_back(opd);
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      auto slice_op =
          rewriter.create<SliceOp>(loc_slice, res.getType(), operands, attrs);
      auto slice_result_var = slice_op.getResult();
      user->eraseOperand(0);
      user->insertOperands(0, slice_result_var);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// slice + slice => slice
struct MergeSlicePattern : public OpRewriterPatternEx<SliceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MergeSlicePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceOp>(context, "MergeSlicePattern") {}

  LogicalResult matchAndRewriteImpl(SliceOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isDynamic()) {
      // SliceOps with dynamic shapes are not allowed to be merged.
      return failure();
    }
    if (!module::isNone(op.getOffsetT()) || !module::isNone(op.getEndsT()) ||
        !module::isNone(op.getStepsT())) {
      return failure();
    }
    auto in_op = op.getInput().getDefiningOp();
    if (!isa<SliceOp>(in_op) || in_op->hasOneUse() == false) {
      return failure();
    }
    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto num_dims = output_shape.size();
    auto in_slice = cast<SliceOp>(in_op);
    auto cur_offset = module::getI64Array(op.getOffset());
    auto cur_ends = module::getI64Array(op.getEnds());
    auto cur_steps = module::getI64Array(op.getSteps());
    auto op_axes = module::getI64Array(op.getHasparamConvertAxesAttr());
    auto in_offset = module::getI64Array(in_slice.getOffset());
    auto in_ends = module::getI64Array(in_slice.getEnds());
    auto in_steps = module::getI64Array(in_slice.getSteps());
    auto in_axes = module::getI64Array(in_slice.getHasparamConvertAxesAttr());
    auto in_axes_num = in_axes->size();
    std::vector<int64_t> axes_ = *op_axes;
    for (int i = 0; i < in_axes_num; ++i) {
      auto axis = in_axes->at(i);
      if (std::find(axes_.begin(), axes_.end(), axis) == axes_.end()) {
        cur_ends->at(axis) = input_shape[axis];
        axes_.push_back(axis);
      }
    }
    std::vector<int64_t> new_offset(num_dims, 0);
    std::vector<int64_t> new_ends(num_dims, 0);
    std::vector<int64_t> new_steps(num_dims, 1);
    for (int i = 0; i < num_dims; i++) {
      auto cur_off = cur_offset->at(i);
      auto cur_end = cur_ends->at(i);
      auto cur_s = cur_steps->at(i);
      assert(cur_s > 0);
      auto in_off = in_offset->at(i);
      auto in_end = in_ends->at(i);
      auto in_s = in_steps->at(i);
      assert(in_s > 0);
      new_offset[i] = in_off + cur_off * in_s;
      new_ends[i] = in_end < cur_end
                        ? in_end
                        : new_offset[i] + (cur_end - cur_off) * in_s;
      new_steps[i] = in_s * cur_s;
    }
    op->setAttr("offset", rewriter.getI64ArrayAttr(new_offset));
    op->setAttr("ends", rewriter.getI64ArrayAttr(new_ends));
    op->setAttr("steps", rewriter.getI64ArrayAttr(new_steps));
    op->setAttr("hasparamConvert_axes", rewriter.getI64ArrayAttr(axes_));
    op->setOperand(0, in_slice.getInput());
    rewriter.eraseOp(in_op);
    return success();
  }
};

struct NoUseSlicePattern : public OpRewriterPatternEx<SliceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  NoUseSlicePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceOp>(context, "NoUseSlicePattern") {}

  LogicalResult matchAndRewriteImpl(SliceOp op,
                                    PatternRewriter &rewriter) const override {

    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    auto steps = module::getI64Array(op.getSteps());
    for (int i = 0; i < in_shape.size(); ++i) {
      if (steps->at(i) == -1)
        return failure();
    }
    if (in_shape.size() != out_shape.size()) {
      return failure();
    }
    for (auto it : llvm::zip(in_shape, out_shape)) {
      if (std::get<0>(it) != std::get<1>(it)) {
        return failure();
      }
    }
    if (!module::isNone(op.getOffsetT()) || !module::isNone(op.getEndsT()) ||
        !module::isNone(op.getStepsT())) {
      return failure();
    }
    op.getOutput().replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

struct TopSliceToReverse : public OpRewriterPatternEx<SliceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopSliceToReverse(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceOp>(context, "TopSliceToReverse") {}

  LogicalResult matchAndRewriteImpl(SliceOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto in_dims = in_shape.size();
    if (in_dims > 4)
      return failure();
    int reverse_count = 0;
    int reverse_dim = 0;
    auto steps = module::getI64Array(op.getSteps());
    for (int i = 0; i < in_dims; i++) {
      if (steps->at(i) == -1 && in_shape[i] == output_shape[i]) {
        reverse_count++;
        reverse_dim = i;
      }
    }
    if (reverse_count != 1)
      return failure();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(reverse_dim)));
    rewriter.replaceOpWithNewOp<ReverseOp>(op, op.getResult().getType(),
                                           op.getInput(), attrs);
    return success();
  }
};

struct TopSliceToGather : public OpRewriterPatternEx<SliceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopSliceToGather(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceOp>(context, "TopSliceToGather") {}

  LogicalResult matchAndRewriteImpl(SliceOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isDynamic()) {
      return failure();
    }
    auto in_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto in_dims = in_shape.size();
    int gather_count = 0;
    int gather_dim = 0;
    auto steps = module::getI64Array(op.getSteps());
    for (int i = 0; i < in_dims; i++) {
      if (steps->at(i) == -1 && in_shape[i] == output_shape[i]) {
        gather_count++;
        gather_dim = i;
      }
    }

    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32()) {
      return failure();
    }
    if (gather_count != 1)
      return failure();

    int dim_length = in_shape[gather_dim];

    std::vector<float> indices;
    for (int i = 0; i < dim_length; i++) {
      indices.push_back(dim_length - i - 1);
    };
    auto coeff_type = RankedTensorType::get(dim_length, rewriter.getF32Type());
    auto indices_op = WeightOp::create(op, "indices", indices, coeff_type);

    // auto indices_op = b ;
    std::vector<NamedAttribute> attrs;
    bool keepdims = true;
    attrs.push_back(
        rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(keepdims)));
    attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(gather_dim)));

    rewriter.replaceOpWithNewOp<GatherOp>(op, op.getResult().getType(),
                                          op.getInput(), indices_op, keepdims,
                                          gather_dim);

    return success();
  }
};

// Some cases can remove Slice , when Conv - Slice.
struct ConvSlice : public OpRewriterPatternEx<SliceOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConvSlice(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceOp>(context, "ConvSlice") {}

  LogicalResult matchAndRewriteImpl(SliceOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_tensor = op.getInput();
    auto in_op = in_tensor.getDefiningOp();
    if (!isa<ConvOp>(in_op))
      return failure();

    // conv' output is used by mult operation is illegal for this patrern.
    if (!in_op->hasOneUse())
      return failure();
    conv_attr_t conv_param = (dyn_cast<ConvOp>(in_op)).parseParam();
    if (conv_param.sh != 1 || conv_param.sw != 1)
      return failure();

    auto Steps = module::getI64Array(op.getSteps());
    auto Offsets = module::getI64Array(op.getOffset());
    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getResult());
    auto conv_out = (dyn_cast<ConvOp>(in_op)).getResult();
    auto conv_oshape = module::getShape(conv_out);
    if (in_shape.size() != 4)
      return failure();
    if (in_shape[1] != out_shape[1])
      return failure();
    if (Steps->at(2) != 1 || Steps->at(3) != 1)
      return failure();

    int crop_h = Offsets->at(2);
    int crop_w = Offsets->at(3);
    if (crop_h > 0 && conv_param.pht < crop_h)
      return failure();
    if (crop_w > 0 && conv_param.pwl < crop_w)
      return failure();

    int crop_h_after = conv_oshape[2] - (out_shape[2] + Offsets->at(2));
    int crop_w_after = conv_oshape[3] - (out_shape[3] + Offsets->at(3));
    if (crop_h_after > 0 && conv_param.phb < crop_h_after)
      return failure();
    if (crop_w_after > 0 && conv_param.pwr < crop_w_after)
      return failure();

    // replace pre op's attr
    conv_param.pht -= crop_h;
    conv_param.phb -= crop_h_after;
    conv_param.pwl -= crop_w;
    conv_param.pwr -= crop_w_after;
    dyn_cast<ConvOp>(in_op).setPadsAttr(rewriter.getI64ArrayAttr(
        {conv_param.pht, conv_param.pwl, conv_param.phb, conv_param.pwr}));
    // get rid of strideslice op
    conv_out.setType(op.getOutput().getType());
    in_op->setLoc(op.getLoc());
    op.getOutput().replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

void SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<NoUseSlicePattern, SplitSlicePattern, MergeSlicePattern,
                 TopSliceToReverse, TopSliceToGather, ConvSlice>(context);
}
