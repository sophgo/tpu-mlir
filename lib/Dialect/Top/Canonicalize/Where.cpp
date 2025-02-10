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

// in gpt2 model, the mask to softmax is from where, very small value in weight
// tensor, change them to -10000
struct FilterWhereWeightPattern : public OpRewriterPatternEx<WhereOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  FilterWhereWeightPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<WhereOp>(context, "FilterWhereWeightPattern") {}

  LogicalResult matchAndRewriteImpl(WhereOp op,
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
      else if ((add_op = dyn_cast<AddOp>(out)))
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
    if (softmax_op == NULL) {
      return failure();
    }
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
// lib/Dialect/Top/Transforms/ProcessorOptimize/OptimizeBM1684X.cpp:expand_dim_and_tile
mlir::Value expand_dim_and_tile(mlir::Value tensor,
                                llvm::ArrayRef<int64_t> out_shape,
                                PatternRewriter &rewriter,
                                std::string op_name) {
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
  std::vector<int64_t> weight_tile(out_dim, 1);

  // tile to expand shape
  int last_i = 0;
  for (int i = out_dim - 1; i >= 0; --i) {
    last_i = i;
    if (tensor_reshape[last_i] != out_shape[last_i]) {
      break;
    }
  }
  int count = 0;
  for (int i = 0; i <= last_i; ++i) {
    if (out_shape[i] == tensor_reshape[i])
      continue;
    int64_t tile = out_shape[i] / tensor_reshape[i];
    weight_tile[i] = tile;
    count++;
  }
  if (count == 0) {
    return tensor_last_op;
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr(weight_tile)));
  auto newType = RankedTensorType::get(out_shape, tensor_stype);
  auto new_name = op_name + "_Tile";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfterValue(tensor_last_op);
  tensor_last_op = rewriter.create<top::TileOp>(
      name_loc, newType, ValueRange{tensor_last_op}, attrs);

  return tensor_last_op;
}

struct WhereBroadcastToTile : public OpRewriterPatternEx<WhereOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  WhereBroadcastToTile(mlir::MLIRContext *context)
      : OpRewriterPatternEx<WhereOp>(context, "WhereBroadcastToTile") {}

  LogicalResult matchAndRewriteImpl(WhereOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isDynamic()) {
      return failure();
    }
    auto out_shape = module::getShape(op.getOutput());
    bool process = false;

    auto cond = op.getCond();
    std::string op_name = module::getName(op.getOutput()).str();
    auto cond_ = expand_dim_and_tile(cond, out_shape, rewriter, op_name);
    if (cond != cond_) {
      op.setOperand(0, cond_);
      process = true;
    }
    return process ? success() : failure();
  }
};

struct WhereTooLarge : public OpRewriterPatternEx<WhereOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  WhereTooLarge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<WhereOp>(context, "WhereTooLarge") {}

  LogicalResult matchAndRewriteImpl(WhereOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op.getResult().hasOneUse())
      return failure();
    if (!op.getXIsConst() && !op.getYIsConst()) {
      return failure();
    }

    auto user = *op.getOutput().getUsers().begin();
    if (isa<AddOp>(user) || isa<ReshapeOp>(user)) {
      Operation *specificOp;
      if (isa<AddOp>(user)) {
        specificOp = dyn_cast<AddOp>(user);
      } else if (isa<ReshapeOp>(user)) {
        specificOp = dyn_cast<ReshapeOp>(user);
      } else {
        return failure();
      }
      if (specificOp->getResult(0).hasOneUse() &&
          isa<SoftmaxOp>(*specificOp->getResult(0).getUsers().begin())) {
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
        if (updated) {
          return success();
        } else {
          return failure();
        }
      }
    }
    return failure();
  }
};

// Where(condition, ConstantFill(x), y) => Where(condition, const, y)
struct WhereFuseConstant : public OpRewriterPatternEx<WhereOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  WhereFuseConstant(mlir::MLIRContext *context)
      : OpRewriterPatternEx<WhereOp>(context, "WhereFuseConstant") {}

  LogicalResult matchAndRewriteImpl(WhereOp op,
                                    PatternRewriter &rewriter) const override {
    auto x_opd = op.getTbrn();
    auto y_opd = op.getFbrn();
    bool fact = false;
    auto none_op = module::getNoneOp(op);
    if (op.getXIsConst()) {
      if (!module::isNone(x_opd)) {
        op.setOperand(1, none_op);
        fact = true;
      }
    } else {
      auto x_op = dyn_cast<top::ConstantFillOp>(x_opd.getDefiningOp());
      if (x_op) {
        op.setXIsConst(true);
        op.setXConstVal(x_op.getValue());
        op.setOperand(1, none_op);
        fact = true;
      }
    }

    if (op.getYIsConst()) {
      if (!module::isNone(y_opd)) {
        op.setOperand(2, none_op);
        fact = true;
      }
    } else {
      auto y_op = dyn_cast<top::ConstantFillOp>(y_opd.getDefiningOp());
      if (y_op) {
        op.setYIsConst(true);
        op.setYConstVal(y_op.getValue());
        op.setOperand(2, none_op);
        fact = true;
      }
    }
    return fact ? success() : failure();
  }
};

/*
  special case in maskrcnn
  where(x!=-1, x, 1)
  x is shape, x == -1 when x < 1
  so, this case can convert to max(x, 1)
*/
struct WhereToMax : public OpRewriterPatternEx<WhereOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  WhereToMax(mlir::MLIRContext *context)
      : OpRewriterPatternEx<WhereOp>(context, "WhereToMax") {}

  LogicalResult matchAndRewriteImpl(WhereOp op,
                                    PatternRewriter &rewriter) const override {
    auto cond = op.getOperand(0);
    // auto x = op.getOperand(1);
    auto y = op.getOperand(2);
    if (module::isShape(y) && op.getXIsConst()) {
      if (auto comp_op = dyn_cast<CompareConstOp>(cond.getDefiningOp())) {
        if (comp_op.getMode().compare("Equal") == 0 &&
            comp_op.getConstVal().convertToDouble() == -1.f &&
            op.getXConstVal().convertToDouble() == 1.f &&
            comp_op->getOperand(0) == y) {

          std::vector<NamedAttribute> attrs;
          attrs.push_back(rewriter.getNamedAttr("const_val",
                                                rewriter.getF64FloatAttr(1.f)));
          auto new_op = rewriter.replaceOpWithNewOp<MaxConstOp>(
              op, op.getOutput().getType(), ValueRange{y}, attrs);
          new_op.shape_inference();
          return success();
        }
      }
    }
    return failure();
  }
};

/*
  special case in VITS:
  remove invalid where

  shape -> Equal(-1) -> where -> nextOP   ==>  shape -> nextOP
        \            /
         ———————————>
*/
struct RemoveInvalidWhere : public OpRewriterPatternEx<WhereOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  RemoveInvalidWhere(mlir::MLIRContext *context)
      : OpRewriterPatternEx<WhereOp>(context, "RemoveInvalidWhere") {}

  LogicalResult matchAndRewriteImpl(WhereOp op,
                                    PatternRewriter &rewriter) const override {
    auto cond = op.getCond();
    auto fbrn = op.getFbrn();

    auto condOp = dyn_cast<CompareConstOp>(cond.getDefiningOp());
    auto fbrnOp = dyn_cast<ShapeOp>(fbrn.getDefiningOp());

    if (!(condOp && fbrnOp)) {
      return failure();
    }

    auto mode = condOp.getMode();
    auto const_val = condOp.getConstVal().convertToDouble();
    if (mode != "Equal" || const_val >= 0) {
      return failure();
    }

    if (!condOp.getResult().hasOneUse()) {
      return failure();
    }

    if (!(op.getOperand(2) == condOp.getOperand())) {
      return failure();
    }

    condOp.getOutput().replaceAllUsesWith(condOp.getInput());
    op.getOutput().replaceAllUsesWith(fbrnOp.getOutput());

    rewriter.eraseOp(condOp);
    rewriter.eraseOp(op);

    return success();
  }
};

void WhereOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<FilterWhereWeightPattern, WhereBroadcastToTile, WhereTooLarge,
                 WhereToMax, RemoveInvalidWhere, WhereFuseConstant>(context);
}
