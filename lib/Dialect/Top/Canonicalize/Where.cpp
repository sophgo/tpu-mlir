//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

// in gpt2 model, the mask to softmax is from where, very small value in weight
// tensor, change them to -10000
struct FilterWhereWeightPattern : public OpRewritePattern<WhereOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhereOp op,
                                PatternRewriter &rewriter) const override {
    if (module::isUniformQuantized(op.getOutput()))
      return failure();
    if (!op->hasOneUse()) {
      return failure();
    }
    int in_cnt = 0;
    int weight_cnt = 0;
    WeightOp weight_op[2] = {NULL};
    SoftmaxOp softmax_op = NULL;
    AddOp add_op = NULL;
    for (auto opd : op.getOperands()) {
      if ((weight_op[weight_cnt] = dyn_cast<WeightOp>(opd.getDefiningOp()))) {
        weight_cnt++;
        if (weight_cnt > 2)
          return failure();
      }
      in_cnt++;
    }
    if (in_cnt != 3 || weight_cnt != 2 || weight_op[0] == NULL ||
        weight_op[1] == NULL) {
      return failure();
    }

    for (auto out : op.getOutput().getUsers()) {
      if ((softmax_op = dyn_cast<SoftmaxOp>(out)))
        break;
      else if (add_op = dyn_cast<AddOp>(out))
        break;
    }
    if (softmax_op == NULL && add_op == NULL)
      return failure();
    else if (add_op != NULL) {
      if (!add_op.getOutput().hasOneUse())
        return failure();
      else if (!isa<SoftmaxOp>(*add_op.getOutput().getUsers().begin()))
        return failure();
      else
        softmax_op =
            dyn_cast<SoftmaxOp>(*add_op.getOutput().getUsers().begin());
    }
    if (softmax_op == NULL)
      return failure();
    for (int i = 0; i < 2; i++) {
      auto w = weight_op[i].read<float>();
      for (int i = 0; i < w.get()->size(); i++) {
        if (w->at(i) < -3e38)
          w->at(i) = -10000;
      }
      const std::vector<float> tmp(*w);
      weight_op[i].update(tmp, (size_t)(w.get()->size()));
    }
    return success();
  }
};

// Idea from Repeat.cpp. Same as
// lib/Dialect/Top/Transforms/ChipOptimize/OptimizeBM1684X.cpp:expand_dim_and_tile
mlir::Value expand_dim_and_tile(mlir::Value tensor,
                                llvm::ArrayRef<int64_t> out_shape,
                                PatternRewriter &rewriter) {
  auto out_dim = out_shape.size();
  auto tensor_shape = module::getShape(tensor);
  auto tensor_dim = tensor_shape.size();
  auto tensor_reshape = shape_expand_dim(tensor_shape, out_dim);
  auto tensor_stype = module::getStorageType(tensor);
  auto tensor_last_op = tensor;
  if (tensor_dim < out_dim) {
    // reshape to out dim
    auto tensorType = RankedTensorType::get(tensor_reshape, tensor_stype);
    std::string in_name = module::getName(tensor).str() + "_ToOutDim";
    auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
    rewriter.setInsertionPointAfterValue(tensor_last_op);
    tensor_last_op = rewriter.create<top::ReshapeOp>(
        loc, tensorType, ValueRange{tensor_last_op});
  }
  auto tensor_last_shape = std::vector<int64_t>(tensor_reshape);

  // tile to expand shape
  int last_i = 0;
  for (int i = out_dim - 1; i >= 0; --i) {
    last_i = i;
    if (tensor_reshape[last_i] != out_shape[last_i]) {
      break;
    }
  }

  for (int i = 0; i <= last_i; ++i) {
    if (out_shape[i] == tensor_reshape[i])
      continue;
    int64_t tile = out_shape[i] / tensor_reshape[i];
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(i)));
    attrs.push_back(
        rewriter.getNamedAttr("tile", rewriter.getI64IntegerAttr(tile)));

    tensor_last_shape[i] = out_shape[i];
    auto newType = RankedTensorType::get(tensor_last_shape, tensor_stype);
    auto new_name =
        module::getName(tensor).str() + "_tile_" + std::to_string(i);
    auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    rewriter.setInsertionPointAfterValue(tensor_last_op);
    tensor_last_op = rewriter.create<top::TileOp>(
        name_loc, newType, ValueRange{tensor_last_op}, attrs);
  }
  return tensor_last_op;
}

struct WhereBroadcastToTile : public OpRewritePattern<WhereOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhereOp op,
                                PatternRewriter &rewriter) const override {
    auto out_shape = module::getShape(op.getOutput());
    bool process = false;

    auto cond = op.getCond();
    auto cond_ = expand_dim_and_tile(cond, out_shape, rewriter);
    if (cond != cond_) {
      op.setOperand(0, cond_);
      process = true;
    }
    return process ? success() : failure();
  }
};

struct WhereTooLarge : public OpRewritePattern<WhereOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhereOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResult().hasOneUse())
      return failure();
    if (!op.getXIsConst() && !op.getYIsConst()) {
      return failure();
    }
    if (isa<AddOp>(*op.getOutput().getUsers().begin())) {
      auto addop = dyn_cast<AddOp>(*op.getOutput().getUsers().begin());
      if (!addop.getOutput().hasOneUse())
        return failure();
      if (isa<SoftmaxOp>(*addop.getOutput().getUsers().begin())) {
        bool updated = false;
        if (op.getXIsConst()) {
          float val = op.getXConstVal().convertToDouble();
          if (val < -3e30) {
            op.setXConstVal(APFloat(-10000.));
            updated = true;
          }
        }
        if (op.getYIsConst()) {
          float val = op.getYConstVal().convertToDouble();
          if (val < -3e30) {
            op.setYConstVal(APFloat(-10000.));
            updated = true;
          }
        }
        if (updated)
          return success();
        else
          return failure();
      }
    }
    return failure();
  }
};

void WhereOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<FilterWhereWeightPattern, WhereBroadcastToTile, WhereTooLarge>(
      context);
}
