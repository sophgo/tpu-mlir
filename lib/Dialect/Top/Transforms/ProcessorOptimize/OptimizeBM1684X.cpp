//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include <future>
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
    int order_size = order->size();
    if (false == (order->at(order_size - 2) == order_size - 1 &&
                  order->at(order_size - 1) == order_size - 2)) {
      return failure();
    }
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

    if (module::isBM1688())
      return failure();
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
      matmul_out2 = is_permute_reshape(matmul0.getRight());
      if (matmul_out2 == NULL)
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
    if (matmul_queries.getInput() == matmul_keys.getInput() &&
        matmul_queries.getInput() != matmul_values.getInput()) {
      return failure();
    }
    auto len = module::getNumElements(matmul_queries.getInput());
    auto n = module::getShape(matmul_queries.getInput())[0];
    auto shape = module::getShape(matmul1.getOutput());
    int64_t head;
    if (shape.size() == 3) {
      head = shape[0] / n;
    } else {
      head = shape[1];
    }
    auto len_weight0 = module::getNumElements(matmul_queries.getRight());
    auto len_weight1 = module::getNumElements(matmul_keys.getRight());
    auto len_weight2 = module::getNumElements(matmul_values.getRight());
    if (module::isBM1688()) {
      // TODO: do not suppose attention when size greater than [batch, 2048,
      // 320]
      if (len / n > 2048 * 320 ||
          (len_weight0 + len_weight1 + len_weight2) > 1024 * 160 * 3) {
        return failure();
      }
    } else if (module::isBM1684X()) {
      if (len / n > 2048 * 320 * 4 ||
          (len_weight0 * 2 + len_weight1 + len_weight2) / head > 1024 * 128 * 4) {
        return failure();
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto none = module::getNoneOp(op);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("scale", mul_const.getConstValAttr()));
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
  auto new_name = module::getName(tensor).str() + "_Tile";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfterValue(tensor_last_op);
  tensor_last_op = rewriter.create<top::TileOp>(
      name_loc, newType, ValueRange{tensor_last_op}, attrs);

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
      } else {
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
      } else {
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

class ConvertConv2DToImg2Col final : public OpRewritePattern<top::ConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ConvOp convOp,
                                PatternRewriter &rewriter) const override {
    Value input = convOp.getInput();
    Value filter = convOp.getFilter();
    Value bias = convOp.getBias();
    Value output = convOp.getOutput();
    auto inputType = llvm::cast<ShapedType>(input.getType());
    auto filterType = llvm::cast<ShapedType>(filter.getType());
    auto outputType = llvm::cast<ShapedType>(output.getType());
    bool with_bias = !module::isNone(bias);
    auto strides = module::getI64Array(convOp.getStrides());
    // note: current support Conv2D
    if (!filterType.hasStaticShape() || !inputType.hasStaticShape() ||
        module::getShape(output).size() != 4) {
      return failure();
    }

    auto hasAllOneValues = [&](mlir::ArrayAttr attr) -> bool {
      return llvm::all_of(
          attr.getAsRange<IntegerAttr>(),
          [](IntegerAttr element) { return element.getInt() == 1; });
    };
    if (convOp.getDilations().has_value() &&
        !hasAllOneValues(convOp.getDilations().value()))
      return failure();

    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();

    const int n = outputShape[0];
    const int oc = outputShape[1];
    const int oh = outputShape[2];
    const int ow = outputShape[3];
    const int ic = filterShape[1];
    const int kh = filterShape[2];
    const int kw = filterShape[3];
    if (!(ic <= 3 && kh >= 16 && kw >= 16 && strides->at(0) == kh &&
          strides->at(1) == kw)) {
      return failure();
    }
    int id = 0;
    auto loc_name = module::getName(convOp.getOperation()).str();
    // 1. Input->Reshape+permute+Reshape(reorder the input)
    SmallVector<int64_t> colTensorShape = {n, ic, oh, kh, ow, kw};
    auto reshapeOp = rewriter.create<top::ReshapeOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get(colTensorShape, inputType.getElementType()),
        ValueRange{input});
    std::vector<int64_t> order = {0, 2, 3, 1, 4, 5};
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));

    auto perMuteOp_0 = rewriter.create<top::PermuteOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oh, kh, ic, ow, kw},
                              inputType.getElementType()),
        ValueRange{reshapeOp}, attrs);
    order = {0, 1, 4, 3, 2, 5};
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    auto perMuteOp = rewriter.create<top::PermuteOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oh, ow, ic, kh, kw},
                              inputType.getElementType()),
        ValueRange{perMuteOp_0}, attrs);

    auto reshapeOp_2 = rewriter.create<top::ReshapeOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oh * ow, ic * kh * kw},
                              inputType.getElementType()),
        ValueRange{perMuteOp});
    // 2. filter->reshape
    auto reshapeOp_3 = rewriter.create<top::ReshapeOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({oc, ic * kh * kw}, filterType.getElementType()),
        ValueRange{filter});
    std::vector<Value> operands;
    operands.emplace_back(reshapeOp_2);
    operands.emplace_back(reshapeOp_3);
    // 3. bias->reshape
    if (with_bias) {
      auto reshapeOp_4 = rewriter.create<top::ReshapeOp>(
          NameLoc::get(
              rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
          RankedTensorType::get(
              {1, 1, oc},
              llvm::cast<ShapedType>(bias.getType()).getElementType()),
          ValueRange{bias});
      operands.emplace_back(reshapeOp_4);
    } else {
      operands.emplace_back(bias);
    }

    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
    attrs.emplace_back(
        rewriter.getNamedAttr("output_transpose", rewriter.getBoolAttr(false)));
    // 4. matmul
    auto matmulOp = rewriter.create<top::MatMulOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType()),
        operands, attrs);
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
    // 5. permute
    auto perMuteOp_2 = rewriter.create<top::PermuteOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType()),
        ValueRange{matmulOp}, attrs);
    // 6. reshape the output
    auto reshapeOp_5 = rewriter.create<top::ReshapeOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oc, oh, ow}, outputType.getElementType()),
        ValueRange{perMuteOp_2});
    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapeOp_5});
    return success();
  }
};

/* for to reduce the data move, split the matmul
   to multiple matmul if match below pattern:
                /--->SliceOp
   MatMul--Reshape(maybe no exist)---->SliceOp
               \---->SliceOp
                \ ---->SliceOp
*/
class SplitMatMulPattern : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Value right = op.getRight();
    Value bias = op.getBias();
    Value output = op.getOutput();
    // auto rightType = llvm::cast<ShapedType>(right.getType());
    auto outputType = llvm::cast<ShapedType>(output.getType());
    auto storage_type = module::getStorageType(right);
    bool with_bias = !module::isNone(bias);
    int32_t id = 0;
    auto loc_name = module::getName(op.getOperation()).str();
    if (!(isa<top::WeightOp>(right.getDefiningOp()) &&
          ((with_bias && isa<top::WeightOp>(bias.getDefiningOp())) ||
           !with_bias))) {
      return failure();
    }

    // fix bug for chatglm2/chatglm3
    if (std::distance(op->user_begin(), op->user_end()) != 0) {
      auto nextOp = *op->user_begin();
      if (isa<top::ReshapeOp>(nextOp)) {
        std::vector<Operation *> users = {nextOp->user_begin(),
                                          nextOp->user_end()};
        if (users.size() == 2 && isa<top::SliceOp>(users[0]) &&
            isa<top::SliceOp>(users[1])) {
          if (!isa<top::ConcatOp>(*users[0]->user_begin())) {
            std::swap(users[0], users[1]);
          }
          if (isa<top::ConcatOp>(*users[0]->user_begin()) &
              isa<top::ReshapeOp>(*users[1]->user_begin()))
            return failure();
        }
      }
    }

    const auto canSplit = [&](Operation *op) {
      using TYPE = std::function<std::pair<bool, std::vector<Operation *>>(
          Operation * op)>;
      TYPE f;
      f = [&](Operation *op) -> decltype(std::declval<TYPE>()(op)) {
        if (std::distance(op->user_begin(), op->user_end()) == 1 &&
            isa<top::ReshapeOp>(*(op->user_begin()))) {
          return f(*(op->user_begin()));
        } else if (std::distance(op->user_begin(), op->user_end()) > 1) {
          std::vector<Operation *> ops;
          for (Operation *user : op->getUsers()) {
            if (!isa<top::SliceOp>(user))
              return std::make_pair(false, std::vector<Operation *>());
            else {
              ops.emplace_back(user);
            }
          }
          return std::make_pair(true, ops);
        } else {
          return std::make_pair(false, std::vector<Operation *>());
        }
      };
      return f(op);
    };

    auto split = canSplit(op.getOperation());
    if (!split.first)
      return failure();
    // current just support weight's col
    auto matmul_shape = module::getShape(output);
    auto right_shape = module::getShape(right);
    llvm::ArrayRef<int64_t> bias_shape;
    int32_t total_slice_width = 0;
    std::vector<int32_t> slice_width;

    // sort the slices ops by offset
    std::packaged_task<std::vector<Operation *>(std::vector<Operation *> &)>
        sortOps([&](std::vector<Operation *> &ops) {
          int32_t index;
          std::vector<Operation *> sorted;
          auto first_offset =
              module::getI64Array(cast<top::SliceOp>(ops[0]).getOffset());
          auto last_offset = module::getI64Array(
              cast<top::SliceOp>(ops[ops.size() - 1]).getOffset());
          for (int32_t i = 0; i < first_offset->size(); i++) {
            if (first_offset->at(i) != last_offset->at(i)) {
              index = i;
              break;
            }
          }

          std::vector<int32_t> offsets;
          for (int32_t i = 0; i < ops.size(); i++) {
            auto offset =
                module::getI64Array(cast<top::SliceOp>(ops[i]).getOffset());
            offsets.push_back(offset->at(index));
          }

          std::vector<int32_t> indexs(ops.size());
          std::iota(indexs.begin(), indexs.end(), 0);
          std::sort(indexs.begin(), indexs.end(),
                    [&](const int32_t &a, const int32_t &b) {
                      return offsets[a] < offsets[b];
                    });
          for (int32_t i = 0; i < ops.size(); i++) {
            sorted.push_back(ops[indexs[i]]);
          }
          return std::move(sorted);
        });

    std::future<std::vector<Operation *>> orderedOps = sortOps.get_future();
    sortOps(split.second);
    std::vector<Operation *> sortedOps = std::move(orderedOps.get());
    for (Operation *user : sortedOps) {
      auto slice_out_shape =
          module::getShape(cast<top::SliceOp>(user).getOutput());
      int32_t w = 1;
      for (int32_t i = matmul_shape.size() - 1; i < slice_out_shape.size(); i++)
        w *= slice_out_shape[i];
      slice_width.push_back(w);
      total_slice_width += w;
    }

    if (matmul_shape[matmul_shape.size() - 1] != total_slice_width)
      return failure();

    int32_t right_first_ndim_size = 1;
    int32_t bias_first_ndim_size = 1;
    for (int32_t i = 0; i < right_shape.size() - 1; i++)
      right_first_ndim_size *= right_shape[i];
    if (with_bias) {
      bias_shape = module::getShape(bias);
      for (int32_t i = 0; i < bias_shape.size() - 1; i++)
        bias_first_ndim_size *= bias_shape[i];
    }

    auto rightOp = cast<top::WeightOp>(right.getDefiningOp());
    auto filter_f32 = rightOp.read_as_float();
    std::shared_ptr<std::vector<float>> bias_f32 = nullptr;
    if (with_bias) {
      auto biasOp = cast<top::WeightOp>(bias.getDefiningOp());
      bias_f32 = biasOp.read_as_float();
    }
    for (auto [idx, value] : llvm::enumerate(sortedOps)) {
      std::vector<Value> operands;
      operands.emplace_back(input);

      auto new_filter_f32 = std::make_shared<std::vector<float>>(
          right_first_ndim_size * slice_width[idx]);
      int32_t offset =
          std::accumulate(slice_width.begin(), slice_width.begin() + idx, 0,
                          std::plus<int32_t>());
      for (int32_t i = 0; i < right_first_ndim_size; i++) {
        for (int32_t k = 0; k < slice_width[idx]; k++)
          new_filter_f32->at(i * slice_width[idx] + k) = filter_f32->at(
              i * right_shape[right_shape.size() - 1] + k + offset);
      }
      SmallVector<int64_t> new_right_shape(right_shape);
      new_right_shape[new_right_shape.size() - 1] = slice_width[idx];
      auto new_right_type =
          RankedTensorType::get(new_right_shape, rewriter.getF32Type()); // rightType.getElementType());
      auto new_filter = top::WeightOp::create(
          op, "_filter_" + std::to_string(id), *new_filter_f32, new_right_type);

      if (storage_type.isF16()) {
        auto new_filter_ = dyn_cast<top::WeightOp>(new_filter.getDefiningOp()).clone_f16(op);
         operands.emplace_back(new_filter_);
      } else {
        operands.emplace_back(new_filter);
      }

      if (with_bias) {
        auto new_bias_f32 = std::make_shared<std::vector<float>>(
            bias_first_ndim_size * slice_width[idx]);
        for (int32_t i = 0; i < bias_first_ndim_size; i++) {
          for (int32_t k = 0; k < slice_width[idx]; k++)
            new_bias_f32->at(i * slice_width[idx] + k) = bias_f32->at(
                i * bias_shape[bias_shape.size() - 1] + k + offset);
        }
        SmallVector<int64_t> new_bias_shape(bias_shape);
        new_bias_shape[new_bias_shape.size() - 1] = slice_width[idx];
        auto new_bias_type = RankedTensorType::get(
            new_bias_shape,
            llvm::cast<ShapedType>(bias.getType()).getElementType());
        auto new_bias = top::WeightOp::create(op, "_bias_" + std::to_string(id),
                                              *new_bias_f32, new_bias_type);
        operands.emplace_back(new_bias);
      } else {
        operands.emplace_back(bias);
      }

      SmallVector<int64_t> new_matmul_shape(matmul_shape);
      new_matmul_shape[new_matmul_shape.size() - 1] = slice_width[idx];
      // rewriter.setInsertionPoint(value);
      auto matmulOp = rewriter.create<top::MatMulOp>(
          NameLoc::get(rewriter.getStringAttr(loc_name + "_matmul_" +
                                              std::to_string(++id))),
          RankedTensorType::get(new_matmul_shape, outputType.getElementType()),
          operands, op->getAttrs());

      if (std::distance(value->user_begin(), value->user_end()) == 1 &&
          isa<top::ReshapeOp, top::SqueezeOp>(*(value->user_begin()))) {
        // trick or temp workaround: op order influence layer group
        auto new_reshape_shape =
            module::getShape((*(value->user_begin()))->getResult(0));
        auto elementType =
            module::getElementType((*(value->user_begin()))->getResult(0));
        auto reshapeOp = rewriter.create<top::ReshapeOp>(
            NameLoc::get(rewriter.getStringAttr(loc_name + "_reshape_" +
                                                std::to_string(id++))),
            RankedTensorType::get(new_reshape_shape, elementType),
            ValueRange{matmulOp});
        rewriter.replaceOp(*(value->user_begin()), reshapeOp);
        rewriter.eraseOp(value);
      } else {
        auto new_reshape_shape =
            module::getShape(cast<top::SliceOp>(value).getOutput());
        if (new_reshape_shape.size() != new_matmul_shape.size()) {
          auto reshapeOp = rewriter.create<top::ReshapeOp>(
              NameLoc::get(rewriter.getStringAttr(loc_name + "_reshape_" +
                                                  std::to_string(id))),
              RankedTensorType::get(new_reshape_shape,
                                    outputType.getElementType()),
              ValueRange{matmulOp});
          rewriter.replaceOp(value, reshapeOp);
        } else {
          rewriter.replaceOp(value, matmulOp);
        }
      }
    }

    return success();
  }
};
} // namespace bm1684x

namespace top {
using namespace bm1684x;
void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns) {
  patterns->add<MergeScale2Conv>(patterns->getContext(), /*PatternBenefit*/ 9);
  patterns
      ->add<ConvertGLMTilePermute, ConvertMatMulWithRightTranspose,
            ConvertMatMul2Attention, ReshapeReorderPattern,
            ConvertMultiInputAdd, WhereBroadcastToTile, ConvertConv2DToImg2Col,
            SplitMatMulPattern, ConvertScaleOp, ConcatToSwapDimInner>(
          patterns->getContext(), 8);
}
} // namespace top
} // namespace tpu_mlir
