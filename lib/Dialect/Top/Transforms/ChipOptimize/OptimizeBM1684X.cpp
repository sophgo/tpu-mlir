//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {

namespace bm1684x {

// Unsqueeze -> Expand(Tile) -> Reshape -> Transpose(Permute) --> MatMul
//                                                       Left -->
// To
// Right -> Transpose(Permute) -> Reshape -> Slice --> MatMul -> Concat
// Left  ->                                  Slice -->
class ConvertGLMTilePermute : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override {

    // 1. match the pattern with
    // Unsqueeze -> Expand(Tile) -> Reshape -> Transpose(Permute) --> MatMul
    auto left = op.getOperand(0); // [32,1,513]
    auto right = op.getOperand(1);
    auto eleType = module::getElementType(left);
    auto right_op = dyn_cast<top::PermuteOp>(right.getDefiningOp());
    if (!right_op) {
      return failure();
    }
    auto reshape_op =
        dyn_cast<top::ReshapeOp>(right_op.getInput().getDefiningOp());
    if (!reshape_op) {
      return failure();
    }
    auto tile_op = dyn_cast<top::TileOp>(reshape_op.getInput().getDefiningOp());
    if (!tile_op) {
      return failure();
    }
    auto unsqueeze_op =
        dyn_cast<top::UnsqueezeOp>(tile_op.getInput().getDefiningOp());
    if (!unsqueeze_op) {
      return failure();
    }
    // not support quant
    if (module::isCalibratedType(op.getOutput().getType())) {
      return failure();
    }

    // 2. Get Params
    auto top = unsqueeze_op.getInput(); // [513,1,2,128]
    auto order = module::getI64Array(right_op.getOrder());
    for (int i = 0; i < order->size(); i++) {
      if (order->at(i) == order->size() - 1) {
        order->at(i) += 1;
      }
    }
    auto op_name = module::getName(op.getOperation()).str();
    std::vector<int64_t> left_shape = module::getShape(left);
    std::vector<int64_t> top_shape = module::getShape(top);
    if (top_shape.size() != 4 || top_shape[2] != 2) {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    auto none_op = module::getNoneOp(op);

    // 3. <Left> SliceOp [32,1,513] -> [16,1,513], [16,1,513]
    attrs.clear();
    operands.clear();
    operands.emplace_back(left);
    operands.emplace_back(none_op);
    operands.emplace_back(none_op);
    operands.emplace_back(none_op);
    attrs.emplace_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(0)));
    attrs.emplace_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr({0, 0, 0})));
    attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1})));
    attrs.emplace_back(rewriter.getNamedAttr(
        "ends", rewriter.getI64ArrayAttr(
                    {left_shape[0] / 2, left_shape[1], left_shape[2]})));
    auto left_slice_type = RankedTensorType::get(
        {left_shape[0] / 2, left_shape[1], left_shape[2]}, eleType);
    auto left_slice_op_0 = rewriter.create<top::SliceOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_left_slice_0")),
        left_slice_type, operands, attrs);

    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(0)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "offset", rewriter.getI64ArrayAttr({left_shape[0] / 2, 0, 0})));
    attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1})));
    attrs.emplace_back(rewriter.getNamedAttr(
        "ends", rewriter.getI64ArrayAttr(
                    {left_shape[0], left_shape[1], left_shape[2]})));
    auto left_slice_op_1 = rewriter.create<top::SliceOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_left_slice_1")),
        left_slice_type, operands, attrs);

    // 4. <Right> PermuteOp [513,1,2,128] -> [2,1,513,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "order", rewriter.getI64ArrayAttr(
                     {2, order->at(0), order->at(1), order->at(2)})));
    auto permute_type = RankedTensorType::get(
        {top_shape[2], top_shape[order->at(0)], top_shape[order->at(1)],
         top_shape[order->at(2)]},
        eleType);
    auto permute_op = rewriter.create<top::PermuteOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_permute")),
        permute_type, top, attrs);

    // 5. <Right> ReshapeOp [2,1,513,128] -> [2,513,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape", rewriter.getI64ArrayAttr(
                     {2, top_shape[order->at(1)], top_shape[order->at(2)]})));
    auto right_reshape_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_reshape")),
        RankedTensorType::get(
            {2, top_shape[order->at(1)], top_shape[order->at(2)]}, eleType),
        permute_op->getResult(0), attrs);

    // 6. <Right> SliceOp [2,513,128] -> [1,513,128], [1,513,128]
    attrs.clear();
    operands.clear();
    operands.emplace_back(right_reshape_op->getResult(0));
    operands.emplace_back(none_op);
    operands.emplace_back(none_op);
    operands.emplace_back(none_op);
    attrs.emplace_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(0)));
    attrs.emplace_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr({0, 0, 0})));
    attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1})));
    attrs.emplace_back(rewriter.getNamedAttr(
        "ends", rewriter.getI64ArrayAttr(
                    {1, top_shape[order->at(1)], top_shape[order->at(2)]})));
    auto right_slice_type = RankedTensorType::get(
        {1, top_shape[order->at(1)], top_shape[order->at(2)]}, eleType);
    auto right_slice_op_0 = rewriter.create<top::SliceOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_right_slice_0")),
        right_slice_type, operands, attrs);

    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(0)));
    attrs.emplace_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr({1, 0, 0})));
    attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1})));
    attrs.emplace_back(rewriter.getNamedAttr(
        "ends", rewriter.getI64ArrayAttr(
                    {2, top_shape[order->at(1)], top_shape[order->at(2)]})));
    auto right_slice_op_1 = rewriter.create<top::SliceOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_right_slice_1")),
        right_slice_type, operands, attrs);

    // 7. MatMulOp [16,1,513] @ [1,513,128] -> [16,1,128]
    attrs.clear();
    operands.clear();
    operands.emplace_back(left_slice_op_0->getResult(0));
    operands.emplace_back(right_slice_op_0->getResult(0));
    operands.emplace_back(none_op);
    auto matmul_type = RankedTensorType::get(
        {left_shape[0] / 2, left_shape[1], top_shape[order->at(2)]}, eleType);
    auto matmul_op_0 = rewriter.create<top::MatMulOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_matmul_0")),
        matmul_type, operands, attrs);

    attrs.clear();
    operands.clear();
    operands.emplace_back(left_slice_op_1->getResult(0));
    operands.emplace_back(right_slice_op_1->getResult(0));
    operands.emplace_back(none_op);
    auto matmul_op_1 = rewriter.create<top::MatMulOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_matmul_1")),
        matmul_type, operands, attrs);

    // 8. ConcatOp [16,1,128],[16,1,128] -> [32,1,128]
    attrs.clear();
    operands.clear();
    operands.emplace_back(matmul_op_0->getResult(0));
    operands.emplace_back(matmul_op_1->getResult(0));
    attrs.emplace_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(0)));
    auto concat_type = RankedTensorType::get(
        {left_shape[0], left_shape[1], top_shape[order->at(2)]}, eleType);
    auto concat_op = rewriter.create<top::ConcatOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_concat")), concat_type,
        operands, attrs);

    // 8. Replace
    rewriter.setInsertionPointAfter(op);
    rewriter.replaceAllUsesWith(op, concat_op->getResult(0));
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertMatMulWithRightTranspose : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    if (module::isWeight(filter)) {
      return failure();
    }
    if (filter.hasOneUse() == false) {
      return failure();
    }
    auto trans_op = dyn_cast<top::PermuteOp>(filter.getDefiningOp());
    if (!trans_op) {
      return failure();
    }
    auto attr = op.parseParam();
    int to_dim = 2;
    if (attr.batch > 1) {
      to_dim = 3;
    }
    std::vector<int64_t> shape = module::getShape(trans_op.getInput());
    auto order = module::getI64Array(trans_op.getOrder());
    std::vector<int64_t> shape_fix;
    std::vector<int64_t> order_fix;
    auto ret = permute_reset(shape, *order, shape_fix, order_fix, to_dim);
    if (ret == false) {
      return failure();
    }
    int n_idx = to_dim - 2;
    int k_idx = to_dim - 1;
    if (shape_fix[n_idx] == attr.N && shape_fix[k_idx] == attr.K &&
        order_fix[n_idx] == k_idx && order_fix[k_idx] == n_idx) {
      // bingo !
      op.setOperand(1, trans_op.getInput());
      op.setRightTranspose(true);
      rewriter.eraseOp(trans_op);
      return success();
    }
    return failure();
  }
};

Value is_reshape_permute(Value in) {
  auto reshape0 = dyn_cast<top::ReshapeOp>(in.getDefiningOp());
  if (!reshape0 || !reshape0->hasOneUse()) {
    return NULL;
  }
  auto permute0 = dyn_cast<top::PermuteOp>(reshape0.getInput().getDefiningOp());
  if (!permute0 || !permute0->hasOneUse()) {
    return NULL;
  }
  auto reshape1 = dyn_cast<top::ReshapeOp>(permute0.getInput().getDefiningOp());
  if (!reshape1) {
    return permute0.getInput();
  } else if (!reshape1->hasOneUse()) {
    return NULL;
  } else {
    return reshape1.getInput();
  }
}

Value is_permute_reshape(Value in) {
  Value permute_out;
  auto reshape0 = dyn_cast<top::ReshapeOp>(in.getDefiningOp());
  if (!reshape0) {
    permute_out = in;
  } else if (!reshape0->hasOneUse()) {
    return NULL;
  } else {
    permute_out = reshape0.getInput();
  }
  auto permute0 = dyn_cast<top::PermuteOp>(permute_out.getDefiningOp());
  if (!permute0 || !permute0->hasOneUse())
    return NULL;
  auto reshape1 = dyn_cast<top::ReshapeOp>(permute0.getInput().getDefiningOp());
  if (!reshape1 || !reshape1->hasOneUse()) {
    return NULL;
  }
  return reshape1.getInput();
}

class ConvertMatMul2Attention : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    // return failure();
    auto filter = op.getRight();
    if (module::isWeight(filter) == false) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }
    Value matmul_out = is_reshape_permute(op.getInput());
    if (matmul_out == NULL) {
      return failure();
    }
    auto matmul1 = dyn_cast<top::MatMulOp>(matmul_out.getDefiningOp());
    if (!matmul1) {
      return failure();
    }
    auto softmax = dyn_cast<top::SoftmaxOp>(matmul1.getInput().getDefiningOp());
    if (!softmax || !softmax->hasOneUse()) {
      return failure();
    }
    Value mul_out;
    auto add = dyn_cast<top::AddOp>(softmax.getInput().getDefiningOp());
    if (!add) {
      mul_out = softmax.getInput();
    } else {
      mul_out = add.getInputs()[0];
    }
    auto mul_const = dyn_cast<top::MulConstOp>(mul_out.getDefiningOp());
    if (!mul_const || !mul_const->hasOneUse()) {
      return failure();
    }
    auto matmul0 =
        dyn_cast<top::MatMulOp>(mul_const.getInput().getDefiningOp());
    if (!matmul0) {
      return failure();
    }
    // queries
    Value matmul_out1 = is_permute_reshape(matmul0.getInput());
    if (matmul_out1 == NULL) {
      return failure();
    }
    auto matmul_queries = dyn_cast<top::MatMulOp>(matmul_out1.getDefiningOp());
    if (!matmul_queries || !module::isWeight(matmul_queries.getRight())) {
      return failure();
    }
    // keys
    auto permute0 =
        dyn_cast<top::PermuteOp>(matmul0.getRight().getDefiningOp());
    if (!permute0 || !permute0->hasOneUse())
      return failure();
    Value matmul_out2 = is_permute_reshape(permute0.getInput());
    if (matmul_out2 == NULL) {
      return failure();
    }
    auto matmul_keys = dyn_cast<top::MatMulOp>(matmul_out2.getDefiningOp());
    if (!matmul_keys || !module::isWeight(matmul_keys.getRight())) {
      return failure();
    }
    // values
    Value matmul_out3 = is_permute_reshape(matmul1.getRight());
    if (matmul_out3 == NULL) {
      return failure();
    }
    auto matmul_values = dyn_cast<top::MatMulOp>(matmul_out3.getDefiningOp());
    if (!matmul_values || !module::isWeight(matmul_values.getRight())) {
      return failure();
    }
    auto len = module::getNumElements(matmul_queries.getInput());
    auto len_weight0 = module::getNumElements(matmul_queries.getRight());
    auto len_weight1 = module::getNumElements(matmul_keys.getRight());
    auto len_weight2 = module::getNumElements(matmul_values.getRight());
    if (module::isBM1686()) {
      // TODO: do not suppose attention when size greater than [batch, 2048,
      // 320]
      if (len > 2048 * 320 ||
          (len_weight0 + len_weight1 + len_weight2) > 1024 * 160 * 3) {
        return failure();
      }
    } else if (module::isBM1684X()) {
      if (len > 2048 * 320 * 4 ||
          (len_weight0 + len_weight1 + len_weight2) > 1024 * 160 * 3 * 4) {
        return failure();
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto none = module::getNoneOp(op);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("scale", mul_const.getConstValAttr()));
    auto batch = module::getShape(op.getOutput())[0];
    auto shape = module::getShape(matmul1.getOutput());
    int64_t head;
    if (shape.size() == 3) {
      head = shape[0] / batch;
    } else {
      head = shape[1];
    }
    attrs.push_back(
        rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(head)));
    if (module::isCalibratedType(op.getOutput().getType())) {
      // quant param
      // qo, ko, vo, m0, si, so, m1
      std::vector<double> scale_v;
      double scale;
      int64_t zp;
      module::getScaleAndZeroPoint(matmul_queries.getOutput(), scale, zp,
                                   false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul_keys.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul_values.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul0.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(mul_const.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(softmax.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul1.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      attrs.push_back(rewriter.getNamedAttr("scale_param",
                                            rewriter.getF64ArrayAttr(scale_v)));
    }
    std::vector<Value> operands;
    operands.push_back(matmul_queries.getInput());
    operands.push_back(matmul_keys.getInput());
    operands.push_back(matmul_values.getInput());
    operands.push_back(matmul_queries.getRight());
    operands.push_back(matmul_queries.getBias());
    operands.push_back(matmul_keys.getRight());
    operands.push_back(matmul_keys.getBias());
    operands.push_back(matmul_values.getRight());
    operands.push_back(matmul_values.getBias());
    operands.push_back(op.getRight());
    operands.push_back(op.getBias());
    operands.push_back(add ? add.getInputs()[1] : none);
    auto attention = rewriter.create<top::AttentionOp>(
        op.getLoc(), op.getOutput().getType(), operands, attrs);
    op.replaceAllUsesWith(attention.getOperation());
    rewriter.eraseOp(op);
    return success();
  }
};

// reorder op when reshapeOp is before matmul/mulconst/cast/softmax op to
// eliminate reshapeOp
class ReshapeReorderPattern : public OpRewritePattern<top::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    auto next_op_ = *output.user_begin();

    if (auto next_op = dyn_cast<top::MatMulOp>(next_op_)) {
      // right is from Reshape too
      auto left = next_op.getInput();
      auto right = next_op.getRight();
      auto right_op_ = right.getDefiningOp();
      auto right_op = dyn_cast<top::ReshapeOp>(right_op_);
      if (op != left.getDefiningOp() || !right_op) {
        return failure();
      }
      // check left and right are both Reshape(n, c, h, w) --> (nxc, h, w)
      auto lshape_ = SmallVector<int64_t>(module::getShape(op.getInput()));
      auto lshape = module::getShape(left);
      if (!(lshape.size() == 3 && lshape_.size() == 4 &&
            lshape[0] == lshape_[0] * lshape_[1] && lshape[1] == lshape_[2] &&
            lshape[2] == lshape_[3])) {
        return failure();
      }
      auto rshape_ = module::getShape(right_op.getInput());
      auto rshape = SmallVector<int64_t>(module::getShape(right));
      if (!(rshape.size() == 3 && rshape_.size() == 4 &&
            rshape[0] == rshape_[0] * rshape_[1] && rshape[1] == rshape_[2] &&
            rshape[2] == rshape_[3])) {
        return failure();
      }
      if (lshape_[0] != rshape_[0] || lshape_[1] != rshape_[1]) {
        return failure();
      }

      // remove left and right ReshapeOp
      op.replaceAllUsesWith(op.getInput());
      right_op.replaceAllUsesWith(right_op.getInput());

      // Update MatMul output shape
      // and update loc to avoid comparing
      auto next_out = next_op.getOutput();
      auto ori_out_type = next_out.getType();
      auto ori_loc = next_op.getLoc();
      auto oshape = module::getShape(next_out);
      std::vector<int64_t> new_oshape{lshape_[0], lshape_[1], oshape[1],
                                      oshape[2]};
      module::setShape(next_out, new_oshape);
      module::setLocSuffix(next_op, "Reshape");

      // Add ReshapeOp after MatMul
      rewriter.setInsertionPointAfterValue(next_out);
      auto new_reshape_op = rewriter.create<top::ReshapeOp>(
          ori_loc, ori_out_type, ValueRange{next_out});
      next_out.replaceAllUsesExcept(new_reshape_op.getOutput(), new_reshape_op);
      rewriter.eraseOp(op);
      rewriter.eraseOp(right_op);
      return success();
    } else if (isa<top::MulConstOp, top::CastOp, top::SoftmaxOp>(next_op_)) {
      // check input is Reshape(n, c, h, w) --> (nxc, h, w)
      auto ishape = SmallVector<int64_t>(module::getShape(op.getInput()));
      auto next_ishape = module::getShape(op.getOutput());
      if (!(next_ishape.size() == 3 && ishape.size() == 4 &&
            next_ishape[0] == ishape[0] * ishape[1] &&
            next_ishape[1] == ishape[2] && next_ishape[2] == ishape[3])) {
        return failure();
      }
      // check next_op param
      if (auto next_op = dyn_cast<top::SoftmaxOp>(next_op_)) {
        int64_t axis = next_op.getAxis();
        if (axis != 2 || axis == -1) {
          return failure();
        }
      }

      // remove ReshapeOp
      op.replaceAllUsesWith(op.getInput());

      // update next_op output shape and modify loc name to avoid comparing
      auto next_out = next_op_->getResult(0);
      auto ori_out_type = next_out.getType();
      auto ori_loc = next_op_->getLoc();
      module::setShape(next_out, ishape);
      module::setLocSuffix(next_op_, "Reshape");

      // Add ReshapeOp after MulConst/Cast/Softmax
      rewriter.setInsertionPointAfterValue(next_out);
      auto new_reshape_op = rewriter.create<top::ReshapeOp>(
          ori_loc, ori_out_type, ValueRange{next_out});
      next_out.replaceAllUsesExcept(new_reshape_op.getOutput(), new_reshape_op);

      if (auto next_op = dyn_cast<top::SoftmaxOp>(next_op_)) {
        next_op->setAttr("axis", rewriter.getSI32IntegerAttr(3));
      }
      rewriter.eraseOp(op);
      return success();
    } else if (auto next_op = dyn_cast<top::ReshapeOp>(next_op_)) {
      auto ishape = module::getShape(op.getInput());
      auto next_oshape = module::getShape(next_op.getOutput());
      if (ishape != next_oshape) {
        return failure();
      }

      op.replaceAllUsesWith(op.getInput());
      next_op.replaceAllUsesWith(next_op.getInput());
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};
class ConvertMultiInputAdd : public OpRewritePattern<top::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(top::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto name = module::getName(op.getOperation()).str();
    if (inputs.size() <= 2) {
      return failure();
    }

    // Start accumulating from the first input
    Value accumulated = inputs[0];
    auto coeffArrayAttr = op.getCoeffAttr().cast<ArrayAttr>();
    for (int i = 1; i < inputs.size(); ++i) {
      Location ori_loc = op.getLoc();
      if (i != inputs.size() - 1) {
        ori_loc =
            NameLoc::get(rewriter.getStringAttr(name + std::to_string(i)));
      }
      auto newCoeffArrayAttr =
          rewriter.getArrayAttr({coeffArrayAttr[i - 1], coeffArrayAttr[i]});
      accumulated = rewriter.create<top::AddOp>(
          ori_loc, accumulated.getType(), ValueRange{accumulated, inputs[i]},
          op.getDoReluAttr(), op.getReluLimitAttr(), newCoeffArrayAttr);
    }

    rewriter.replaceOp(op, accumulated);
    return success();
  }
};

class ConvertScaleOp : public OpRewritePattern<top::ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    if (input_shape.size() > 4) {
      return failure();
    }
    auto cur_scale = dyn_cast<top::WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias = dyn_cast<top::WeightOp>(op.getBias().getDefiningOp());
    if (!(cur_scale && cur_bias) || input_shape.size() < 3) {
      return failure();
    }
    int channel = cur_scale.getType().cast<RankedTensorType>().getNumElements();
    auto cur_scale_f32 = cur_scale.read<float>();
    auto cur_bias_f32 = cur_bias.read<float>();

    std::vector<float> new_scale_v(channel);
    std::vector<float> new_bias_v(channel);
    std::copy(cur_scale_f32->begin(), cur_scale_f32->end(),
              new_scale_v.begin());
    std::copy(cur_bias_f32->begin(), cur_bias_f32->end(), new_bias_v.begin());

    // scale to depthwise convolution
    NamedAttrList attrs;
    attrs.set("kernel_shape", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("strides", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0}));
    attrs.set("group", rewriter.getI64IntegerAttr(channel));
    attrs.set("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    auto relu_limit = op.getReluLimit().convertToDouble();
    attrs.set("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    auto filter_type =
        RankedTensorType::get({channel, 1, 1, 1}, rewriter.getF32Type());
    auto new_scale =
        top::WeightOp::create(op, "_to_weight", new_scale_v, filter_type);
    auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
    auto new_bias =
        top::WeightOp::create(op, "_to_bias", new_bias_v, bias_type);

    rewriter.replaceOpWithNewOp<top::ConvOp>(
        op, op.getResult().getType(),
        ValueRange{op.getInput(), new_scale, new_bias}, attrs);
    return success();
  }
};

class MergeScale2Conv : public OpRewritePattern<top::ScaleOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<top::ConvOp>(formerOp)) {
      return failure();
    }
    auto conv_op = cast<top::ConvOp>(formerOp);
    if (conv_op.getDoRelu()) {
      return failure();
    }

    auto cur_scale_op = dyn_cast<top::WeightOp>(op.getScale().getDefiningOp());
    auto cur_bias_op = dyn_cast<top::WeightOp>(op.getBias().getDefiningOp());
    auto cur_scale_f32 = cur_scale_op.read<float>();
    auto cur_bias_f32 = cur_bias_op.read<float>();

    auto conv_weight_op =
        dyn_cast<top::WeightOp>(conv_op.getFilter().getDefiningOp());
    auto conv_bias_op =
        dyn_cast<top::WeightOp>(conv_op.getBias().getDefiningOp());
    if (!conv_weight_op || !conv_bias_op) {
      return failure();
    }

    int64_t oc, ic, kh, kw;
    module::getNCHW(conv_weight_op.getOutput(), oc, ic, kh, kw);

    // merge weight: weight = weight * cur_scale
    std::vector<float> conv_weight_v(oc * ic * kh * kw, 0);
    auto conv_weight_f32 = conv_weight_op.read<float>();
    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < kw * kh * ic; ++j) {
        conv_weight_v[i * ic * kh * kw + j] =
            conv_weight_f32->at(i * ic * kh * kw + j) * cur_scale_f32->at(i);
      }
    }
    // merge bias: bias = bias * cur_scale + cur_bias
    std::vector<float> conv_bias_v(oc, 0);
    if (conv_bias_op != nullptr) {
      auto conv_bias_f32 = conv_bias_op.read<float>();
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] =
            conv_bias_f32->at(i) * cur_scale_f32->at(i) + cur_bias_f32->at(i);
      }
    } else {
      for (int i = 0; i < oc; ++i) {
        conv_bias_v[i] = cur_bias_f32->at(i);
      }
    }

    auto weight_type =
        RankedTensorType::get({oc, ic, kh, kw}, rewriter.getF32Type());
    auto conv_weight = top::WeightOp::create(conv_op, "merged_to_conv_weight",
                                             conv_weight_v, weight_type);
    auto bias_type = RankedTensorType::get({oc}, rewriter.getF32Type());
    auto conv_bias = top::WeightOp::create(conv_op, "merged_to_conv_bias",
                                           conv_bias_v, bias_type);
    conv_op->setOperand(1, conv_weight);
    conv_op->setOperand(2, conv_bias);
    conv_op.getOutput().setType(op.getOutput().getType());
    // update attrs
    double relu_limit = op.getReluLimit().convertToDouble();
    formerOp->setLoc(op.getLoc());
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    formerOp->setAttr("relu_limit", rewriter.getF64FloatAttr(relu_limit));

    // remove the scale Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

// Same as lib/Dialect/Top/Canonicalize/Where.cpp:expand_dim_and_tile
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
    // reshape to out
    std::string in_name = module::getName(tensor).str() + "_ToOutDim";
    auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
    rewriter.setInsertionPointAfterValue(tensor_last_op);
    if (tensor.getType()
            .cast<RankedTensorType>()
            .getElementType()
            .isa<quant::CalibratedQuantizedType>()) {
      auto i_type = module::getCalibratedType(tensor);
      auto tensorType = RankedTensorType::get(tensor_reshape, i_type);
      tensor_last_op = rewriter.create<top::ReshapeOp>(
          loc, tensorType, ValueRange{tensor_last_op});
    } else {
      auto tensorType = RankedTensorType::get(tensor_reshape, tensor_stype);
      tensor_last_op = rewriter.create<top::ReshapeOp>(
          loc, tensorType, ValueRange{tensor_last_op});
    }
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

class WhereBroadcastToTile : public OpRewritePattern<top::WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::WhereOp op,
                                PatternRewriter &rewriter) const override {
    auto out_shape = module::getShape(op.getOutput());
    bool process = false;

    if (!op.getXIsConst()) {
      auto tbrn = op.getTbrn();
      if (isa<top::WeightOp>(tbrn.getDefiningOp()) &&
          tbrn.getType().cast<RankedTensorType>().getNumElements() == 1) {
        // single value in tensor, set to const
        float value = dyn_cast<top::WeightOp>(tbrn.getDefiningOp())
                          .read_as_float()
                          ->at(0);
        op.setXConstValAttr(rewriter.getF64FloatAttr(value));
        op.setXIsConst(true);
        auto fbrn = op.getFbrn();
        op.setOperand(1, module::getNoneOp(op));
        op.setOperand(2, fbrn);
        process = true;
      } else if (!isa<top::WeightOp>(tbrn.getDefiningOp())) {
        auto tbrn_ = expand_dim_and_tile(tbrn, out_shape, rewriter);
        if (tbrn != tbrn_) {
          op.setOperand(1, tbrn_);
          process = true;
        }
      }
    }
    if (!op.getYIsConst()) {
      auto fbrn = op.getFbrn();
      if (isa<top::WeightOp>(fbrn.getDefiningOp()) &&
          fbrn.getType().cast<RankedTensorType>().getNumElements() == 1) {
        // single value in tensor, set to const
        float value = dyn_cast<top::WeightOp>(fbrn.getDefiningOp())
                          .read_as_float()
                          ->at(0);
        op.setYConstValAttr(rewriter.getF64FloatAttr(value));
        op.setYIsConst(true);
        auto tbrn = op.getTbrn();
        op.setOperand(1, tbrn);
        op.setOperand(2, module::getNoneOp(op));
        process |= true;
      } else if (!isa<top::WeightOp>(fbrn.getDefiningOp())) {
        auto fbrn_ = expand_dim_and_tile(fbrn, out_shape, rewriter);
        if (fbrn != fbrn_) {
          op.setOperand(2, fbrn_);
          process |= true;
        }
      }
    }

    return process ? success() : failure();
  }
};

} // namespace bm1684x

namespace top {
using namespace bm1684x;
void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      ConvertGLMTilePermute,
      ConvertMatMulWithRightTranspose,
      ConvertMatMul2Attention,
      ReshapeReorderPattern,
      MergeScale2Conv,
      ConvertScaleOp,
      ConvertMultiInputAdd,
      WhereBroadcastToTile
  >(patterns->getContext());
  // clang-format on
}
} // namespace top
} // namespace tpu_mlir
