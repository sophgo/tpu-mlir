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

#define DEBUG_TYPE "Arg_canonicalize"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

typedef sg_reduce_method_t Reduce_method;
Reduce_method reduce_max = SG_REDUCE_MAX;
Reduce_method reduce_min = SG_REDUCE_MIN;
typedef arg_method_t Arg_method;
Arg_method arg_max = ARG_MAXT;
Arg_method arg_min = ARG_MINT;
struct TopArgReducefull : public OpRewriterPatternEx<ArgOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopArgReducefull(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ArgOp>(context, "TopArgReducefull") {}

  LogicalResult matchAndRewriteImpl(ArgOp op,
                                    PatternRewriter &rewriter) const override {
    auto formerOp = op.getInput().getDefiningOp();
    if (!op->hasOneUse() || formerOp->hasOneUse()) {
      return failure();
    }
    auto arg_method = llvm::StringSwitch<int>(op.getMode())
                          .Case("ArgMax", 0)
                          .Case("ArgMin", 1)
                          .Default(-1);
    auto arg_axis = op.getAxis();
    bool match = false;
    auto reduce_method_exp = (arg_method == arg_max) ? reduce_max : reduce_min;
    for (auto &use : formerOp->getUses()) {
      if (use.getOwner() == op)
        continue;
      if (!isa<ReduceOp>(use.getOwner()))
        continue;
      auto reop = dyn_cast<top::ReduceOp>(use.getOwner());
      auto reduce_method = llvm::StringSwitch<int>(reop.getMode())
                               .Case("ReduceMax", 2)
                               .Case("ReduceMin", 3)
                               .Default(-1);
      if (reduce_method != reduce_method_exp)
        continue;
      auto reduce_axes = module::getI64Array(reop.getAxes());
      if (reduce_axes->size() != 1)
        continue;
      if (reduce_axes->at(0) != arg_axis)
        continue;
      match = true;
      auto no_values = op.getValues().getType().isa<NoneType>();
      auto reop_out_shape = module::getShape(reop.getOutput());
      auto reop_out_type = module::getStorageType(reop.getOutput());
      auto new_type = RankedTensorType::get(reop_out_shape, reop_out_type);
      op.getValues().setType(new_type);
      if (no_values) {
        std::vector<Location> locs_v = {op.getIndices().getLoc()};
        std::string out_values_name =
            module::getName(reop.getOutput()).str() + "_r_values";
        auto values_loc = NameLoc::get(rewriter.getStringAttr(out_values_name));
        locs_v.push_back(values_loc);
        module::setLoc(op.getValues(), values_loc);
        auto fused_loc = FusedLoc::get(getContext(), locs_v);
        op->setLoc(fused_loc);
      }
      reop.replaceAllUsesWith(op.getValues());
      if (reop.getOutput().getUsers().empty()) {
        rewriter.eraseOp(reop);
      }
    }
    if (!match)
      return failure();
    return success();
  }
};

struct TopTransposeArg : public OpRewriterPatternEx<ArgOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopTransposeArg(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ArgOp>(context, "TopTransposeArg") {}

  LogicalResult matchAndRewriteImpl(ArgOp op,
                                    PatternRewriter &rewriter) const override {

    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<PermuteOp>(formerOp)) {
      return failure();
    }
    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getIndices());
    auto permuteOp = cast<PermuteOp>(formerOp);
    auto old_axis = op.getAxis();
    auto permute_order = module::getI64Array(permuteOp.getOrder());
    auto permute_order_len = permute_order->size();
    int order_mask[permute_order_len - 1];
    auto Keepdim = op.getKeepdims();
    memset(order_mask, 0, sizeof(int) * (permute_order_len - 1));
    int order_dim = 0;
    for (int i = 0; i < permute_order_len; i++) {
      if (i == old_axis)
        continue;
      order_mask[order_dim++] = permute_order->at(i);
    }
    for (int i = 0; i < permute_order_len - 2; i++) {
      if (order_mask[i] < order_mask[i + 1])
        continue;
      return failure();
    }
    auto arg_axis = permute_order->at(old_axis);
    op->setAttr("axis", rewriter.getI64IntegerAttr(arg_axis));
    op->setOperand(0, permuteOp.getInput());
    std::vector<int64_t> out_shape(output_shape.size(), 0);
    int out_dim_count = 0;
    for (int i = 0; i < input_shape.size(); i++) {
      if (i == old_axis) {
        if (!Keepdim)
          continue;
        out_shape[out_dim_count] = 1;
        out_dim_count += 1;
      } else {
        out_shape[out_dim_count] = input_shape[i];
        out_dim_count += 1;
      }
    }
    // reshape of arg.indices
    auto out_indices_type = module::getStorageType(op.getIndices());
    auto new_indices_type = RankedTensorType::get(out_shape, out_indices_type);
    std::string out_indices_name =
        module::getName(op.getIndices()).str() + "_r_Reshape";
    auto indices_loc = NameLoc::get(rewriter.getStringAttr(out_indices_name));
    rewriter.setInsertionPointAfter(op);
    auto rs_indices_op = rewriter.create<ReshapeOp>(
        indices_loc, new_indices_type, ValueRange{op.getIndices()});
    op.getIndices().replaceAllUsesExcept(rs_indices_op.getOutput(),
                                         rs_indices_op);
    // reshape of arg.values
    auto values_flag = op.getValues().getType().isa<NoneType>();
    if (!values_flag) {
      auto out_values_type = module::getStorageType(op.getValues());
      auto new_values_type = RankedTensorType::get(out_shape, out_values_type);
      std::string out_values_name =
          module::getName(op.getValues()).str() + "_r_Reshape";
      auto values_loc = NameLoc::get(rewriter.getStringAttr(out_values_name));
      rewriter.setInsertionPointAfter(op);
      auto rs_values_op = rewriter.create<ReshapeOp>(
          values_loc, new_values_type, ValueRange{op.getValues()});
      op.getValues().replaceAllUsesExcept(rs_values_op.getOutput(),
                                          rs_values_op);
    }
    if (permuteOp.getOutput().getUsers().empty()) {
      rewriter.eraseOp(permuteOp);
    }
    return success();
  }
};

void ArgOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<TopArgReducefull>(context, "TopArgReducefull");
  results.insert<TopTransposeArg>(context, "TopTransposeArg");
}
