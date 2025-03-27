//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct SplitToSlice : public OpRewriterPatternEx<SplitOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  SplitToSlice(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SplitOp>(context, "SplitToSlice") {}

  LogicalResult matchAndRewriteImpl(SplitOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_shape = module::getShape(op.getInput());
    int64_t dims = in_shape.size();
    auto num = op.getNum();
    auto axis = op.getAxis();
    std::vector<int64_t> offset(dims, 0);
    std::vector<int64_t> steps(dims, 1);
    std::vector<int64_t> ends(dims, std::numeric_limits<int64_t>::max());
    // auto name = module::getName(op.getResult(0)).str();
    ends[axis] = 0;
    rewriter.setInsertionPointAfter(op);
    for (int i = 0; i < num; i++) {
      auto name = module::getName(op.getResult(i)).str();
      auto out = op.getResult(i);
      auto out_shape = module::getShape(out);
      ends[axis] += out_shape[axis];
      //   auto out_name = name + "_tpu_" + std::to_string(i);
      //   auto name_loc = NameLoc::get(rewriter.getStringAttr(out_name));
      auto name_loc = NameLoc::get(rewriter.getStringAttr(name));
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset)));
      attrs.push_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
      attrs.push_back(
          rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(ends)));
      attrs.push_back(rewriter.getNamedAttr("hasparamConvert_axes",
                                            rewriter.getI64ArrayAttr(axis)));
      auto none = module::getNoneOp(op);
      std::vector<Value> operands;
      const auto &opd = op->getOperand(0);
      operands.push_back(opd);
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      auto s_op =
          rewriter.create<SliceOp>(name_loc, out.getType(), operands, attrs);
      out.replaceAllUsesWith(s_op.getOutput());
      offset[axis] += out_shape[axis];
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct SplitReshape2ReshapeSplit : public OpRewriterPatternEx<SplitOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  SplitReshape2ReshapeSplit(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewriterPatternEx<SplitOp>(context, "SplitReshape2ReshapeSplit",
                                     benefit) {}

  LogicalResult matchAndRewriteImpl(SplitOp op,
                                    PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto in_shape = module::getShape(input);
    int64_t dims = in_shape.size();
    auto axis = op.getAxis();
    if (axis < 0)
      axis += dims;
    llvm::ArrayRef<int64_t> last_RO_shape;
    int axis_shape = 0, new_axis = -1;
    for (auto out : op.getResults()) {
      auto RI_shape = module::getShape(out);
      int64_t Ndim_RI = RI_shape.size();
      auto out_op = out.user_begin();
      if (!out.hasOneUse() || !isa<ReshapeOp>(*out_op)) {
        return failure();
      }
      auto reshape_out = dyn_cast<ReshapeOp>(*out_op).getResult();
      auto RO_shape = module::getShape(reshape_out);
      int64_t Ndim_RO = RO_shape.size();
      new_axis = axis + Ndim_RO - Ndim_RI;
      if (out == *(op.getResults().begin()))
        last_RO_shape = RO_shape;
      else {
        for (int64_t i = 0; i < Ndim_RO; i++) {
          if (RO_shape[i] != last_RO_shape[i] && i != new_axis)
            return failure();
        }
      }
      if (new_axis >= 0 && new_axis < Ndim_RI) {
        axis_shape += RO_shape[new_axis];
      }
    }
    if (axis_shape != in_shape[axis])
      return failure();
    // change split attribute
    // add one reshapeOP before split
    rewriter.setInsertionPointAfterValue(input);

    std::string name = module::getName(input).str();
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_r_reshape"));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> shape;
    for (auto v : last_RO_shape) {
      shape.push_back(v);
    }
    shape[new_axis] = in_shape[axis];
    attrs.emplace_back(
        rewriter.getNamedAttr("shape", rewriter.getI64ArrayAttr(shape)));
    // Attention!!!!
    // when create OP, the Type is return Type. And Type is not only data
    // type(int, float..), it contains shape info.
    auto reshape_otype =
        RankedTensorType::get(shape, module::getElementType(input));
    auto reshape_op = rewriter.create<top::ReshapeOp>(loc, reshape_otype,
                                                      ValueRange{input}, attrs);

    attrs.clear();
    std::vector<Location> split_locs_v;
    for (auto out : op.getResults()) {
      ReshapeOp rop = dyn_cast<ReshapeOp>(*out.user_begin());
      split_locs_v.push_back(rop.getLoc());
    }
    auto split_loc = FusedLoc::get(getContext(), split_locs_v);
    attrs.push_back(
        rewriter.getNamedAttr("num", rewriter.getI64IntegerAttr(op.getNum())));
    attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(new_axis)));
    auto split_op = rewriter.create<top::SplitOp>(
        split_loc, op.getOutputs().getType(), reshape_op.getOutput(), attrs);
    op.replaceAllUsesWith(split_op.getOperation());

    // delete all reshapeOP aftered split
    // Attention!!! after replaceAllUsesWith the op has been delete. so use
    // split_op
    for (auto out : split_op.getResults()) {
      ReshapeOp rop = dyn_cast<ReshapeOp>(*out.user_begin());
      // Attention !!!
      // if not setType, will cause ERROR such like "doesn't match function
      // result type"
      out.setType(rop.getOutput().getType());
      rop.getOutput().replaceAllUsesWith(rop.getInput());
      rewriter.eraseOp(rop);
    }
    return success();
  }
};

void SplitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<SplitToSlice, SplitReshape2ReshapeSplit>(context);
}
