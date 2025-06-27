//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "tpu_mlir/Conversion/TopToTpu/TopToTpu.h"
#include <future>
namespace tpu_mlir {

namespace bm1684x {

// Unsqueeze -> Expand(Tile) -> Reshape -> Transpose(Permute) --> MatMul
//                                                       Left -->
// To
// Right -> Transpose(Permute) -> Reshape -> Slice --> MatMul -> Concat
// Left  ->                                  Slice -->
class ConvertGLMTilePermute : public OpRewriterPatternEx<top::MatMulOp> {
public:
  ConvertGLMTilePermute(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MatMulOp>(context, "ConvertGLMTilePermute") {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MatMulOp op,
                      mlir::PatternRewriter &rewriter) const override {
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
    if (!unsqueeze_op || module::getShape(unsqueeze_op.getInput())[1] != 1) {
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

// Unsqueeze -> Expand(Tile) -> Reshape -> Transpose(Permute) --> MatMul
//                                                    Reshape --> Left -->
// To
// Right -> Transpose(Permute) -> Reshape --> MatMul (do loop batch times)
//                               Left  ->
// support multi batch
class ConvertGLMTilePermute2 : public OpRewriterPatternEx<top::MatMulOp> {
public:
  ConvertGLMTilePermute2(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MatMulOp>(context, "ConvertGLMTilePermute2") {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MatMulOp op,
                      mlir::PatternRewriter &rewriter) const override {
    // [2x32x512x512;512x2x2x128] -> 2x[16x512x2x512;1x512x2x128]
    // (hdim_is_batch)
    // 1. match the pattern with
    // Unsqueeze -> Expand(Tile) -> Reshape -> Transpose(Permute) --> MatMul
    // TODO support case [512x64x512;512x4x128]; eliminate all permute ops
    auto left = op.getOperand(0);  // [64x512x512](64 = batch*num_heads)
    auto right = op.getOperand(1); // []
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
    auto top = unsqueeze_op.getInput(); // [512,2,2,128]
    auto order = module::getI64Array(right_op.getOrder());
    if (order->size() != 3)
      return failure();
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
    int batch_size = top_shape[1];
    int query_group = top_shape[2];
    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    auto none_op = module::getNoneOp(op);

    // 3. <right> PermuteOp [512,2,2,128] -> [2,2,512,128]
    attrs.clear();
    operands.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "order", rewriter.getI64ArrayAttr(
                     {order->at(0), 2, order->at(1), order->at(2)})));
    auto right_permute_type = RankedTensorType::get(
        {top_shape[order->at(0)], top_shape[2], top_shape[order->at(1)],
         top_shape[order->at(2)]},
        eleType);
    auto right_permute_op = rewriter.create<top::PermuteOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_right_permute")),
        right_permute_type, top, attrs);

    // 4. <Right> ReshapeOp [2,2,512,128] -> [4,512,128]
    std::vector<int64_t> right_reshape_shape = {batch_size * query_group,
                                                top_shape[order->at(1)],
                                                top_shape[order->at(2)]};
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape", rewriter.getI64ArrayAttr({batch_size * query_group,
                                           top_shape[order->at(1)],
                                           top_shape[order->at(2)]})));
    auto right_reshape_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_right_reshape")),
        RankedTensorType::get({batch_size * query_group,
                               top_shape[order->at(1)],
                               top_shape[order->at(2)]},
                              eleType),
        right_permute_op->getResult(0), attrs);

    int slice_sec = batch_size * query_group;
    int left_slice_size = left_shape[0] / slice_sec;
    std::vector<Value> matmul_operands;
    std::vector<Value> concat_operands;
    for (int slice_idex = 0; slice_idex < slice_sec; slice_idex++) {
      // 6. <Left>  SliceOp [64, 512, 512] -> 4 [16, 512, 512]
      attrs.clear();
      operands.clear();
      operands.emplace_back(left);
      operands.emplace_back(none_op);
      operands.emplace_back(none_op);
      operands.emplace_back(none_op);
      attrs.emplace_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(0)));
      attrs.emplace_back(rewriter.getNamedAttr(
          "offset",
          rewriter.getI64ArrayAttr({left_slice_size * slice_idex, 0, 0})));
      attrs.emplace_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1})));
      attrs.emplace_back(rewriter.getNamedAttr(
          "ends", rewriter.getI64ArrayAttr({left_slice_size * (slice_idex + 1),
                                            left_shape[1], left_shape[2]})));
      auto left_slice_type = RankedTensorType::get(
          {left_slice_size, left_shape[1], left_shape[2]}, eleType);
      auto left_slice_op = rewriter.create<top::SliceOp>(
          NameLoc::get(rewriter.getStringAttr(op_name + "_left_slice_" +
                                              std::to_string(slice_idex))),
          left_slice_type, operands, attrs);
      matmul_operands.emplace_back(left_slice_op->getResult(0));

      // 7. <Right>  SliceOp [4, 512, 128] -> 4 [1, 512, 128]
      attrs.clear();
      operands.clear();
      operands.emplace_back(right_reshape_op);
      operands.emplace_back(none_op);
      operands.emplace_back(none_op);
      operands.emplace_back(none_op);
      attrs.emplace_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(0)));
      attrs.emplace_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr({slice_idex, 0, 0})));
      attrs.emplace_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1})));
      attrs.emplace_back(rewriter.getNamedAttr(
          "ends",
          rewriter.getI64ArrayAttr({slice_idex + 1, right_reshape_shape[1],
                                    right_reshape_shape[2]})));
      auto right_slice_type = RankedTensorType::get(
          {1, right_reshape_shape[1], right_reshape_shape[2]}, eleType);
      auto right_slice_op = rewriter.create<top::SliceOp>(
          NameLoc::get(rewriter.getStringAttr(op_name + "_right_slice_" +
                                              std::to_string(slice_idex))),
          right_slice_type, operands, attrs);
      matmul_operands.emplace_back(right_slice_op->getResult(0));
    }
    // do matmul after all sliceops to enabled to layergroup
    // 8. MatMulOp [16, 512, 512] -> [1, 512, 128]
    for (int slice_idex = 0; slice_idex < slice_sec; slice_idex++) {
      attrs.clear();
      operands.clear();
      operands.emplace_back(matmul_operands[slice_idex * 2]);
      operands.emplace_back(matmul_operands[slice_idex * 2 + 1]);
      operands.emplace_back(none_op);
      auto matmul_type =
          RankedTensorType::get({left_shape[0] / (batch_size * query_group),
                                 left_shape[1], right_reshape_shape[2]},
                                eleType);
      auto matmul_op = rewriter.create<top::MatMulOp>(
          NameLoc::get(rewriter.getStringAttr(op_name + "_matmul_" +
                                              std::to_string(slice_idex))),
          matmul_type, operands, attrs);
      concat_operands.emplace_back(matmul_op->getResult(0));
    }
    // 9. ConcatOp [16, 512, 128] -> [batch_size * query_group * 16, 512, 128]
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(0)));
    auto concat_type = RankedTensorType::get(
        {left_shape[0], left_shape[1], right_reshape_shape[2]}, eleType);
    auto concat_op = rewriter.create<top::ConcatOp>(
        NameLoc::get(rewriter.getStringAttr(op_name + "_concat")), concat_type,
        concat_operands, attrs);

    rewriter.setInsertionPointAfter(op);
    rewriter.replaceAllUsesWith(op, concat_op->getResult(0));
    rewriter.eraseOp(op);
    return success();
  }
};

class ChatGLM3ToGQAAttention : public OpRewriterPatternEx<top::MatMulOp> {
public:
  ChatGLM3ToGQAAttention(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MatMulOp>(context, "ChatGLM3ToGQAAttention") {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MatMulOp op,
                      mlir::PatternRewriter &rewriter) const override {
    // Rewrite to match GQA FlashAttention
    auto matmul_left = op.getOperand(0);
    auto matmul_right = op.getOperand(1);
    auto reshape0_op = dyn_cast<top::ReshapeOp>(matmul_left.getDefiningOp());
    if (!reshape0_op || !module::isWeight(matmul_right)) {
      return failure();
    }
    auto transpose_op =
        dyn_cast<top::PermuteOp>(reshape0_op.getInput().getDefiningOp());
    if (!transpose_op) {
      return failure();
    }
    auto reshape1_op =
        dyn_cast<top::ReshapeOp>(transpose_op.getInput().getDefiningOp());
    if (!reshape1_op) {
      return failure();
    }
    auto matmulkv_op =
        dyn_cast<top::MatMulOp>(reshape1_op.getInput().getDefiningOp());
    if (!matmulkv_op) {
      return failure();
    }
    // not support quant
    if (module::isCalibratedType(op.getOutput().getType())) {
      return failure();
    }

    auto reshape0_kv_left_op =
        dyn_cast<top::ReshapeOp>(matmulkv_op.getInput().getDefiningOp());
    if (!reshape0_kv_left_op) {
      return failure();
    }
    auto transpose_kv_right_op =
        dyn_cast<top::PermuteOp>(matmulkv_op.getRight().getDefiningOp());
    if (!transpose_kv_right_op) {
      return failure();
    }
    auto reshape_kv_right_op = dyn_cast<top::ReshapeOp>(
        transpose_kv_right_op.getInput().getDefiningOp());
    if (!reshape_kv_right_op) {
      return failure();
    }
    auto tile_kv_right_op =
        dyn_cast<top::TileOp>(reshape_kv_right_op.getInput().getDefiningOp());
    if (!tile_kv_right_op) {
      return failure();
    }
    auto unsqueeze_kv_right_op =
        dyn_cast<top::UnsqueezeOp>(tile_kv_right_op.getInput().getDefiningOp());
    if (!unsqueeze_kv_right_op) {
      return failure();
    }

    Operation *concat_kv_right_op;
    if (isa<top::ConcatOp>(unsqueeze_kv_right_op.getInput().getDefiningOp()))
      concat_kv_right_op = dyn_cast<top::ConcatOp>(
          unsqueeze_kv_right_op.getInput().getDefiningOp());
    else if (isa<top::ReshapeOp>(
                 unsqueeze_kv_right_op.getInput().getDefiningOp()))
      concat_kv_right_op = dyn_cast<top::ReshapeOp>(
          unsqueeze_kv_right_op.getInput().getDefiningOp());
    else
      return failure();
    auto softmax_op = dyn_cast<top::SoftmaxOp>(
        reshape0_kv_left_op.getInput().getDefiningOp());
    if (!softmax_op) {
      return failure();
    }
    auto add_op = dyn_cast<top::AddOp>(softmax_op.getInput().getDefiningOp());
    if (!add_op) {
      return failure();
    }
    auto reshape1_kv_left_op =
        dyn_cast<top::ReshapeOp>(add_op.getInputs()[0].getDefiningOp());
    if (!reshape1_kv_left_op) {
      return failure();
    }
    auto mulconst_op = dyn_cast<top::MulConstOp>(
        reshape1_kv_left_op.getInput().getDefiningOp());
    if (!mulconst_op) {
      return failure();
    }

    auto matmulqk_op =
        dyn_cast<top::MatMulOp>(mulconst_op.getInput().getDefiningOp());
    if (!matmulqk_op) {
      return failure();
    }
    auto transpose_qk_left_op =
        dyn_cast<top::PermuteOp>(matmulqk_op.getInput().getDefiningOp());
    if (!transpose_qk_left_op) {
      return failure();
    }
    auto reshape_qk_left_op = dyn_cast<top::ReshapeOp>(
        transpose_qk_left_op.getInput().getDefiningOp());
    if (!reshape_qk_left_op) {
      return failure();
    }
    auto concat_qk_left_op =
        dyn_cast<top::ConcatOp>(reshape_qk_left_op.getInput().getDefiningOp());
    if (!concat_qk_left_op) {
      return failure();
    }
    auto transpose_qk_right_op =
        dyn_cast<top::PermuteOp>(matmulqk_op.getRight().getDefiningOp());
    if (!transpose_qk_left_op) {
      return failure();
    }
    auto reshape_qk_right_op = dyn_cast<top::ReshapeOp>(
        transpose_qk_right_op.getInput().getDefiningOp());
    if (!reshape_qk_right_op) {
      return failure();
    }
    auto tile_qk_right_op =
        dyn_cast<top::TileOp>(reshape_qk_right_op.getInput().getDefiningOp());
    if (!tile_qk_right_op) {
      return failure();
    }
    auto unsqueeze_qk_right_op =
        dyn_cast<top::UnsqueezeOp>(tile_qk_right_op.getInput().getDefiningOp());
    if (!unsqueeze_qk_right_op) {
      return failure();
    }
    auto concat_qk_right_op = dyn_cast<top::ConcatOp>(
        unsqueeze_qk_right_op.getInput().getDefiningOp());
    if (!concat_qk_right_op) {
      return failure();
    }
    auto qk_left_shape = module::getShape(concat_qk_left_op.getOutput());
    auto qk_right_shape = module::getShape(tile_qk_right_op.getOutput());
    if (qk_left_shape.size() != 4 || qk_left_shape[1] != 1)
      return failure();

    // Reshape[16k,1,32,128] -> [1,16k,32,128]
    std::vector<NamedAttribute> attrs;
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape",
        rewriter.getI64ArrayAttr({qk_left_shape[1], qk_left_shape[0],
                                  qk_left_shape[2], qk_left_shape[3]})));
    auto qk_left_reshape_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_qk_left_op.getOperation()).str() +
            "_reshape")),
        RankedTensorType::get(
            {qk_left_shape[1], qk_left_shape[0], qk_left_shape[2],
             qk_left_shape[3]},
            module::getElementType(concat_qk_left_op.getOutput())),
        concat_qk_left_op.getOutput(), attrs);

    // PermuteOp [1,16k,32,128] -> [1,32,1k,128]
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
    auto qk_left_permute_type = RankedTensorType::get(
        {qk_left_shape[1], qk_left_shape[2], qk_left_shape[0],
         qk_left_shape[3]},
        module::getElementType(qk_left_reshape_op.getOutput()));
    auto qk_left_permute_op = rewriter.create<top::PermuteOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_qk_left_op.getOperation()).str() +
            "_permute")),
        qk_left_permute_type, qk_left_reshape_op.getOutput(), attrs);

    // ReshapeOp [16k,1,2,128] -> [1,16k,2,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape",
        rewriter.getI64ArrayAttr({qk_right_shape[1], qk_right_shape[0],
                                  qk_right_shape[2], qk_right_shape[4]})));
    auto qk_right_reshape0_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_qk_right_op.getOperation()).str() +
            "_reshape0")),
        RankedTensorType::get(
            {qk_right_shape[1], qk_right_shape[0], qk_right_shape[2],
             qk_right_shape[4]},
            module::getElementType(concat_qk_right_op.getOutput())),
        concat_qk_right_op.getOutput(), attrs);

    // UnsqueezeOp [1,16k,2,128] -> [1,16k,2,1,168]
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({3})));
    auto qk_right_unsqueeze_type = RankedTensorType::get(
        {qk_right_shape[1], qk_right_shape[0], qk_right_shape[2], 1,
         qk_right_shape[4]},
        module::getElementType(qk_right_reshape0_op.getOutput()));
    auto qk_right_unsqueeze_op = rewriter.create<top::UnsqueezeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_qk_right_op.getOperation()).str() +
            "_unsqueeze")),
        qk_right_unsqueeze_type, qk_right_reshape0_op.getOutput(), attrs);

    // TileOp [1,16k,2,1,128] -> [1,16k,2,16,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "tile", rewriter.getI64ArrayAttr({1, 1, 1, qk_right_shape[3], 1})));
    auto qk_right_tile_type = RankedTensorType::get(
        {qk_right_shape[1], qk_right_shape[0], qk_right_shape[2],
         qk_right_shape[3], qk_right_shape[4]},
        module::getElementType(qk_right_unsqueeze_op.getOutput()));
    auto qk_right_tile_op = rewriter.create<top::TileOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_qk_right_op.getOperation()).str() +
            "_tile")),
        qk_right_tile_type, ValueRange{qk_right_unsqueeze_op.getOutput()},
        attrs);

    // ReshapeOp [1,16k,2,16,128] -> [1,16k,32,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape",
        rewriter.getI64ArrayAttr({qk_right_shape[1], qk_right_shape[0],
                                  qk_right_shape[2] * qk_right_shape[3],
                                  qk_right_shape[4]})));
    auto qk_right_reshape1_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_qk_right_op.getOperation()).str() +
            "_reshape1")),
        RankedTensorType::get(
            {qk_right_shape[1], qk_right_shape[0],
             qk_right_shape[2] * qk_right_shape[3], qk_right_shape[4]},
            module::getElementType(concat_qk_right_op.getOutput())),
        qk_right_tile_op.getOutput(), attrs);

    // PermuteOp [1,16k,32,128] -> [1,32,128,16k]
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 3, 1})));
    auto qk_right_permute_type = RankedTensorType::get(
        {qk_right_shape[1], qk_right_shape[2] * qk_right_shape[3],
         qk_right_shape[4], qk_right_shape[0]},
        module::getElementType(concat_qk_right_op.getOutput()));
    auto qk_right_permute_op = rewriter.create<top::PermuteOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_qk_right_op.getOperation()).str() +
            "_permute")),
        qk_right_permute_type, qk_right_reshape1_op.getOutput(), attrs);

    // MatMulOp [(1,32,16k,128),(1,32,128,16k)] -> [1,32,16k,16k]
    std::vector<int64_t> new_matmulqk_shape =
        module::getShape(qk_left_permute_op.getOutput());
    new_matmulqk_shape[new_matmulqk_shape.size() - 1] =
        module::getShape(qk_right_permute_op.getOutput())[3];
    auto new_matmulqk_op = rewriter.clone(*matmulqk_op);
    module::setLocSuffix(new_matmulqk_op, std::to_string(0));
    new_matmulqk_op->setOperand(0, qk_left_permute_op.getOutput());
    new_matmulqk_op->setOperand(1, qk_right_permute_op.getOutput());

    new_matmulqk_op->setAttr(
        "right_transpose",
        rewriter.getBoolAttr(matmulqk_op.getRightTranspose()));
    new_matmulqk_op->setAttr("hdim_is_batch", rewriter.getBoolAttr(false));
    module::setShape(new_matmulqk_op->getResult(0), new_matmulqk_shape);

    // MulConstOp [1,32,16k,16k]
    auto new_mulconst_op = rewriter.clone(*mulconst_op);
    module::setLocSuffix(new_mulconst_op, std::to_string(0));
    new_mulconst_op->setOperand(0, new_matmulqk_op->getResult(0));
    std::vector<int64_t> new_mulconst_shape =
        module::getShape(new_matmulqk_op->getResult(0));
    module::setShape(new_mulconst_op->getResult(0), new_mulconst_shape);

    // AddOp [1,32,16k,16k]
    auto new_add_op = rewriter.clone(*add_op);
    module::setLocSuffix(new_add_op, std::to_string(0));
    new_add_op->setOperand(0, new_mulconst_op->getResult(0));
    new_add_op->setOperand(1, add_op.getInputs()[1]);
    std::vector<int64_t> new_add_shape =
        module::getShape(new_mulconst_op->getResult(0));
    module::setShape(new_add_op->getResult(0), new_add_shape);

    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "axis", rewriter.getSI32IntegerAttr(new_matmulqk_shape.size() - 1)));
    auto softmax_name_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(softmax_op.getOperation()).str() + "_" +
        std::to_string(0)));
    auto new_softmax_op = rewriter.create<top::SoftmaxOp>(
        softmax_name_loc,
        RankedTensorType::get(new_matmulqk_shape,
                              module::getElementType(softmax_op.getOutput())),
        ValueRange{new_add_op->getResult(0)}, attrs);

    auto kv_right_shape = module::getShape(tile_kv_right_op.getOutput());
    // ReshapeOp [16k,1,2,128] -> [1,16k,2,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape",
        rewriter.getI64ArrayAttr({kv_right_shape[1], kv_right_shape[0],
                                  kv_right_shape[2], kv_right_shape[4]})));
    auto kv_right_reshape0_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_kv_right_op).str() + "_reshape0")),
        RankedTensorType::get(
            {kv_right_shape[1], kv_right_shape[0], kv_right_shape[2],
             kv_right_shape[4]},
            module::getElementType(concat_kv_right_op->getResult(0))),
        concat_kv_right_op->getResult(0), attrs);

    // UnsqueezeOp [1,16k,2,128] -> [1,16k,2,1,168]
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({3})));
    auto kv_right_unsqueeze_type = RankedTensorType::get(
        {kv_right_shape[1], kv_right_shape[0], kv_right_shape[2], 1,
         kv_right_shape[4]},
        module::getElementType(kv_right_reshape0_op.getOutput()));
    auto kv_right_unsqueeze_op = rewriter.create<top::UnsqueezeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_kv_right_op).str() + "_unsqueeze")),
        kv_right_unsqueeze_type, kv_right_reshape0_op.getOutput(), attrs);

    // TileOp [1,16k,2,1,128] -> [1,16k,2,16,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "tile", rewriter.getI64ArrayAttr({1, 1, 1, kv_right_shape[3], 1})));
    auto kv_right_tile_type = RankedTensorType::get(
        {kv_right_shape[1], kv_right_shape[0], kv_right_shape[2],
         kv_right_shape[3], kv_right_shape[4]},
        module::getElementType(qk_right_unsqueeze_op.getOutput()));
    auto kv_right_tile_op = rewriter.create<top::TileOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_kv_right_op).str() + "_tile")),
        kv_right_tile_type, ValueRange{kv_right_unsqueeze_op.getOutput()},
        attrs);

    // ReshapeOp [1,16k,2,16,128] -> [1,16k,32,128]
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape",
        rewriter.getI64ArrayAttr({kv_right_shape[1], kv_right_shape[0],
                                  kv_right_shape[2] * kv_right_shape[3],
                                  kv_right_shape[4]})));
    auto kv_right_reshape1_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_kv_right_op).str() + "_reshape1")),
        RankedTensorType::get(
            {kv_right_shape[1], kv_right_shape[0],
             kv_right_shape[2] * kv_right_shape[3], kv_right_shape[4]},
            module::getElementType(concat_kv_right_op->getResult(0))),
        kv_right_tile_op.getOutput(), attrs);

    // PermuteOp [1,16k,32,128] -> [1,32,16k,128]
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
    auto kv_right_permute_type = RankedTensorType::get(
        {kv_right_shape[1], kv_right_shape[2] * kv_right_shape[3],
         kv_right_shape[0], kv_right_shape[4]},
        module::getElementType(concat_kv_right_op->getResult(0)));
    auto kv_right_permute_op = rewriter.create<top::PermuteOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(concat_kv_right_op).str() + "_permute")),
        kv_right_permute_type, kv_right_reshape1_op.getOutput(), attrs);

    // MatMulOp [(1,32,16k,16k),(1,32,16k,128)] -> [1,32,16k,128]
    std::vector<int64_t> new_matmulkv_shape =
        module::getShape(new_softmax_op.getOutput());
    new_matmulkv_shape[new_matmulkv_shape.size() - 1] =
        module::getShape(kv_right_permute_op.getOutput())[3];
    auto new_matmulkv_op = rewriter.clone(*matmulkv_op);
    module::setLocSuffix(new_matmulkv_op, std::to_string(0));
    new_matmulkv_op->setOperand(0, new_softmax_op.getOutput());
    new_matmulkv_op->setOperand(1, kv_right_permute_op.getOutput());
    new_matmulkv_op->setAttr("hdim_is_batch", rewriter.getBoolAttr(false));
    module::setShape(new_matmulkv_op->getResult(0), new_matmulkv_shape);

    // PermuteOp [1,32,16k,128] -> [1,16k,32,128]
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3})));
    auto matmulkv_permute_type = RankedTensorType::get(
        {new_matmulkv_shape[0], new_matmulkv_shape[2], new_matmulkv_shape[1],
         new_matmulkv_shape[3]},
        module::getElementType(new_matmulkv_op->getResult(0)));
    auto matmulkv_permute_op = rewriter.create<top::PermuteOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(new_matmulkv_op).str() + "_permute")),
        matmulkv_permute_type, new_matmulkv_op->getResult(0), attrs);

    // ReshapeOp [1,16k,32,128] -> [16k,1,4096]
    attrs.clear();
    auto attention_shape = module::getShape(matmulkv_op.getOutput());
    std::vector<int64_t> new_attention_shape = attention_shape;
    new_attention_shape[0] = attention_shape[1];
    new_attention_shape[1] = new_matmulkv_shape[0];
    new_attention_shape[2] = new_matmulkv_shape[1] * new_matmulkv_shape[3];
    attrs.emplace_back(rewriter.getNamedAttr(
        "shape", rewriter.getI64ArrayAttr({new_attention_shape[0],
                                           new_attention_shape[1],
                                           new_attention_shape[2]})));
    auto attention_reshape_op = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(new_matmulkv_op).str() + "_reshape")),
        RankedTensorType::get(
            {new_attention_shape[0], new_attention_shape[1],
             new_attention_shape[2]},
            module::getElementType(new_matmulkv_op->getResult(0))),
        matmulkv_permute_op->getResult(0), attrs);

    op->setOperand(0, attention_reshape_op.getOutput());

    rewriter.setInsertionPointAfter(op);
    return success();
  }
};

class ConvertMatMulWithRightTranspose
    : public OpRewriterPatternEx<top::MatMulOp> {
public:
  ConvertMatMulWithRightTranspose(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MatMulOp>(
            context, "ConvertMatMulWithRightTranspose", benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MatMulOp op,
                      mlir::PatternRewriter &rewriter) const override {
    if (module::isSG2380()) {
      if (module::Mode::F32 == module::getMode()) {
        // sg2380 can't support GDMA transpose instruction in FP32 mode
        return failure();
      }
    }
    auto filter = op.getRight();
    if (module::isWeight(filter)) {
      return failure();
    }

    // find the cascading matmul op using the filter other than the current op
    std::vector<top::MatMulOp> matmul_ops = getCascadingMatMulOp(filter, op);
    if (filter.hasOneUse() == false && matmul_ops.size() == 0) {
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
      for (auto matmul_op : matmul_ops) {
        matmul_op.setRightTranspose(!matmul_op.getRightTranspose());
        matmul_op.setOperand(1, trans_op.getInput());
      }
      op.setRightTranspose(!op.getRightTranspose());
      rewriter.eraseOp(trans_op);
      return success();
    }
    return failure();
  }

private:
  std::vector<top::MatMulOp> getCascadingMatMulOp(Value in,
                                                  top::MatMulOp op) const {
    std::vector<top::MatMulOp> matmul_ops;
    auto permute = dyn_cast<top::PermuteOp>(in.getDefiningOp());
    if (!permute) {
      return std::vector<top::MatMulOp>();
    }
    for (auto user : permute->getUsers()) {
      auto matmul = dyn_cast<top::MatMulOp>(user);
      if (!matmul || matmul.getRight() != permute->getResult(0)) {
        return std::vector<top::MatMulOp>();
      }
      if (matmul == op) {
        continue;
      } else {
        matmul_ops.push_back(matmul);
      }
    }
    return matmul_ops;
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

class ConvertMatMul2Attention : public OpRewriterPatternEx<top::MatMulOp> {
public:
  ConvertMatMul2Attention(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MatMulOp>(context, "ConvertMatMul2Attention",
                                           benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MatMulOp op,
                      mlir::PatternRewriter &rewriter) const override {
    if (module::isBM1688() || module::isBM1690Family() || module::isSG2380() ||
        module::isMARS3() || module::isSGTPUV8()) {
      return failure();
    }
    if (module::isHighPrecision()) {
      return failure();
    }
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
      auto shape = module::getShape(add.getInputs()[1]);
      auto batch = module::getShape(op.getInput())[0];
      auto ele_num = module::getNumElements(add.getInputs()[1]);
      if (batch * shape[shape.size() - 1] != ele_num) {
        return failure();
      }
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
    auto queries_shape = module::getShape(matmul_queries.getInput());
    auto keys_shape = module::getShape(matmul_keys.getInput());
    auto values_shape = module::getShape(matmul_values.getInput());
    if (queries_shape[0] != keys_shape[0] ||
        queries_shape[0] != values_shape[0]) {
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
    if (module::isBM1688() || module::isMARS3() || module::isSGTPUV8()) {
      // TODO: do not suppose attention when size greater than [batch, 2048,
      // 320]
      if (len / n > 2048 * 320 ||
          (len_weight0 + len_weight1 + len_weight2) > 1024 * 160 * 3) {
        return failure();
      }
    } else if (module::isBM1684X()) {
      if (len / n > 2048 * 320 * 4 ||
          (len_weight0 * 2 + len_weight1 + len_weight2) / head >
              1024 * 128 * 4) {
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
class ReshapeReorderPattern : public OpRewriterPatternEx<top::ReshapeOp> {
public:
  ReshapeReorderPattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ReshapeOp>(context, "ReshapeReorderPattern",
                                            benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::ReshapeOp op,
                      mlir::PatternRewriter &rewriter) const override {
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

class ConvertMultiInputAdd : public OpRewriterPatternEx<top::AddOp> {
public:
  ConvertMultiInputAdd(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::AddOp>(context, "ConvertMultiInputAdd",
                                        benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::AddOp op,
                      mlir::PatternRewriter &rewriter) const override {
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
          op.getDoReluAttr(), op.getReluLimitAttr(), newCoeffArrayAttr,
          op.getIsScalarAttr());
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

class WhereBroadcastToTile : public OpRewriterPatternEx<top::WhereOp> {
public:
  WhereBroadcastToTile(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::WhereOp>(context, "WhereBroadcastToTile",
                                          benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::WhereOp op,
                      mlir::PatternRewriter &rewriter) const override {
    if (module::isDynamic()) {
      return failure();
    }
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

class ConvertConv2DToImg2Col : public OpRewriterPatternEx<top::ConvOp> {
public:
  ConvertConv2DToImg2Col(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ConvOp>(context, "ConvertConv2DToImg2Col",
                                         benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::ConvOp convOp,
                      mlir::PatternRewriter &rewriter) const override {
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
    auto inputShape = inputType.getShape();

    const int n = outputShape[0];
    const int ih = inputShape[2];
    const int iw = inputShape[3];
    const int oc = outputShape[1];
    const int oh = outputShape[2];
    const int ow = outputShape[3];
    const int ic = filterShape[1];
    const int kh = filterShape[2];
    const int kw = filterShape[3];
    if (!(ic <= 3 && kh >= 14 && kw >= 14 && strides->at(0) == kh &&
          strides->at(1) == kw)) {
      return failure();
    }
    if (ih % kh != 0 || iw % kw != 0) {
      return failure();
    }
    // When kh >= 29 and kw >= 29, the last dimension of the reordered kernel
    // becomes quite large. Using it as the right matrix in matrix
    // multiplication, particularly when performing a transpose on the right
    // matrix, can lead to performance degradation.This adjustment is primarily
    // made concerning the CLIP model.Further improvements will be considered
    // later
    if ((module::isBM1688() || module::isSGTPUV8() || module::isSG2380() ||
         module::isMARS3()) &&
        !(kh < 29 && kw < 29)) {
      return failure();
    }
    if (module::isSG2380() && module::getMode() == module::Mode::F32) {
      // sg2380 can't support GDMA transpose instruction in FP32 mode
      return failure();
    }
    int id = 0;
    auto loc_name = module::getName(convOp.getOperation()).str();
    // 1. Input->Reshape+permute+Reshape(reorder the input)
    SmallVector<int64_t> colTensorShape = {n, ic, oh, kh, ow, kw};
    LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                               std::to_string(id));
    auto reshapeOp = rewriter.create<top::ReshapeOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get(colTensorShape, inputType.getElementType()),
        ValueRange{input});
    std::vector<int64_t> order = {0, 2, 3, 1, 4, 5};
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));

    LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                               std::to_string(id));
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
    LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                               std::to_string(id));
    auto perMuteOp = rewriter.create<top::PermuteOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oh, ow, ic, kh, kw},
                              inputType.getElementType()),
        ValueRange{perMuteOp_0}, attrs);
    LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                               std::to_string(id));
    auto reshapeOp_2 = rewriter.create<top::ReshapeOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oh * ow, ic * kh * kw},
                              inputType.getElementType()),
        ValueRange{perMuteOp});
    // 2. filter->reshape
    LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                               std::to_string(id));
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
      LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                                 std::to_string(id));
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
    LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                               std::to_string(id));
    auto matmulOp = rewriter.create<top::MatMulOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType()),
        operands, attrs);
    attrs.clear();
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 2, 1})));
    // 5. permute
    LoweringConfig::split_map[loc_name].insert(loc_name + "_" +
                                               std::to_string(id));
    auto perMuteOp_2 = rewriter.create<top::PermuteOp>(
        NameLoc::get(
            rewriter.getStringAttr(loc_name + "_" + std::to_string(id++))),
        RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType()),
        ValueRange{matmulOp}, attrs);
    // 6. reshape the output, keep the name as output
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(
        convOp, convOp.getOutput().getType(), ValueRange{perMuteOp_2});
    return success();
  }
};

// A tensor's requantOp should be close with the tensor's producer (eq.
// MatMulOP)
class ForwardRequantInt : public OpRewriterPatternEx<top::RequantIntOp> {
public:
  ForwardRequantInt(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::RequantIntOp>(context, "ForwardRequantInt",
                                               benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::RequantIntOp requantIntOp,
                      mlir::PatternRewriter &rewriter) const override {
    auto formerOp = requantIntOp.getInput().getDefiningOp();
    if (!isa<top::PermuteOp>(formerOp) && !isa<top::ReshapeOp>(formerOp) ||
        !formerOp->hasOneUse()) {
      return failure();
    }
    auto axis = requantIntOp.getRqAxis();
    if (axis < 0) {
      axis += module::getShape(requantIntOp.getInput()).size();
    }
    if (auto permuteOp = dyn_cast<top::PermuteOp>(formerOp)) {
      auto permute_ishape = module::getShape(permuteOp.getInput());
      auto permute_order = module::getI64Array(permuteOp.getOrder());
      int32_t new_rq_axis = permute_order->at(axis);
      // Now, requantOp with rq_axis=-1 is only valid when it would be fused
      // into a MatMulOp
      if (new_rq_axis == -1 ||
          new_rq_axis == module::getShape(requantIntOp.getInput()).size() - 1) {
        Operation *former_op = permuteOp.getInput().getDefiningOp();
        while (true) {
          if (isa<top::MatMulOp>(former_op))
            break; // valid case
          if (isa<top::PermuteOp, top::ReshapeOp>(former_op)) {
            former_op = former_op->getOperands()[0].getDefiningOp();
            if (former_op->hasOneUse())
              continue;
          }
          return failure();
        }
      }
      auto newRequantIntOp = rewriter.create<top::RequantIntOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(permuteOp.getInput()).str() + "_requanted")),
          module::getTypeLike(requantIntOp.getOutput(), permute_ishape),
          permuteOp.getInput(), requantIntOp->getAttrs());
      if (new_rq_axis == -1 ||
          new_rq_axis == module::getShape(requantIntOp.getInput()).size() - 1) {
        newRequantIntOp->setAttr("fuse_rq", rewriter.getBoolAttr(true));
      }
      newRequantIntOp->setAttr("rq_axis",
                               rewriter.getSI32IntegerAttr(new_rq_axis));
      auto newPermuteOp = rewriter.create<top::PermuteOp>(
          requantIntOp.getLoc(), requantIntOp.getOutput().getType(),
          newRequantIntOp.getOutput(), permuteOp->getAttrs());
      rewriter.replaceAllUsesWith(requantIntOp.getOutput(),
                                  newPermuteOp.getOutput());
      rewriter.eraseOp(requantIntOp);
      rewriter.eraseOp(permuteOp);
      return success();
    } else if (auto reshapeOp = dyn_cast<top::ReshapeOp>(formerOp)) {
      auto reshape_ishape = module::getShape(reshapeOp.getInput());
      auto reshape_oshape = module::getShape(reshapeOp.getOutput());
      for (int i = 0; i < axis; ++i) {
        if (reshape_ishape[i] != reshape_oshape[i]) {
          return failure();
        }
      }
      auto newRequantIntOp = rewriter.create<top::RequantIntOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(reshapeOp.getInput()).str() + "_requanted")),
          module::getTypeLike(requantIntOp.getOutput(), reshape_ishape),
          reshapeOp.getInput(), requantIntOp->getAttrs());

      // requantInt has differnt logic for rq_axis = -1 and rq_axis != -1
      auto raw_shift = module::getI64Array(requantIntOp.getRshift());
      for (size_t idx = 0; idx < raw_shift->size(); ++idx) {
        raw_shift->at(idx) = -raw_shift->at(idx);
      }
      newRequantIntOp->setAttr("rshift", rewriter.getI64ArrayAttr(*raw_shift));

      auto newReshapeOp = rewriter.create<top::ReshapeOp>(
          requantIntOp.getLoc(), requantIntOp.getOutput().getType(),
          newRequantIntOp.getOutput(), reshapeOp->getAttrs());
      rewriter.replaceAllUsesWith(requantIntOp.getOutput(),
                                  newReshapeOp.getOutput());
      rewriter.eraseOp(requantIntOp);
      rewriter.eraseOp(reshapeOp);
      return success();
    }
    return failure();
  }
};

/* for to reduce the data move, split the matmul
   to multiple matmul if match below pattern:
                /--->SliceOp
   MatMul--Reshape(maybe no exist)---->SliceOp
               \---->SliceOp
                \ ---->SliceOp
*/
class SplitMatMulPattern : public OpRewriterPatternEx<top::MatMulOp> {
public:
  SplitMatMulPattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MatMulOp>(context, "SplitMatMulPattern",
                                           benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MatMulOp op,
                      mlir::PatternRewriter &rewriter) const override {
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
        // check all users are SliceOp
        for (auto user : users) {
          if (!isa<top::SliceOp>(*user))
            return failure();
        }
        if (users.size() == 2) {
          // what does this mean?
          if (users[0]->user_begin() == users[0]->user_end() ||
              users[1]->user_begin() == users[1]->user_end()) {
            return failure();
          }
          if (!isa<top::ConcatOp>(*users[0]->user_begin())) {
            std::swap(users[0], users[1]);
          }
          if (isa<top::ConcatOp>(*users[0]->user_begin()) &
              isa<top::ReshapeOp>(*users[1]->user_begin())) {
            return failure();
          }
        }
        // fix bug for internv1 while slicing matmul col not at first dim
        // equivalently case0 : reshape(1x2400x6144->1x2400x8x6x128) +
        // slice(1x2400x8x4x128) : invalid case1 :
        // reshape(64x49x288->64x49x3x3x32) + slice(64x49x1x3x32) : valid
        // not strict judgement, find the first diff axis of matmul col;
        // slice.axes may be empty, judge slice axis by shape
        // real slice.axes size is 1 mostly
        auto reshape_input_shape = module::getShape(nextOp->getOperands()[0]);
        for (auto user : users) {
          auto slice_op = dyn_cast<top::SliceOp>(*user);
          auto slice_input_shape = module::getShape(slice_op.getInput());
          auto slice_output_shape = module::getShape(slice_op.getOutput());
          int slice_input_dim = slice_input_shape.size();
          int last_diff_axis = slice_input_dim - 1;
          int reshape_outer_size = 0;
          for (; reshape_input_shape[reshape_outer_size] ==
                 slice_input_shape[reshape_outer_size];
               reshape_outer_size++) {
          }
          for (; last_diff_axis >= reshape_outer_size; last_diff_axis--) {
            if (slice_input_shape[last_diff_axis] !=
                    slice_output_shape[last_diff_axis] &&
                std::max((last_diff_axis == reshape_outer_size)
                             ? 1
                             : slice_input_shape[last_diff_axis - 1],
                         (last_diff_axis == reshape_outer_size)
                             ? 1
                             : slice_output_shape[last_diff_axis - 1]) > 1)
              break;
          }
          if (last_diff_axis >= reshape_outer_size)
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
            auto inshape =
                module::getShape(cast<top::SliceOp>(ops[i]).getInput());
            if (offset->at(index) >= 0) {
              offsets.push_back(offset->at(index));
            } else {
              offsets.push_back(offset->at(index) + inshape[index]);
            }
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
      std::vector<int64_t> new_right_shape(right_shape);
      new_right_shape[new_right_shape.size() - 1] = slice_width[idx];
      auto new_filter = top::WeightOp::create_float(
          op, "_filter_" + std::to_string(id), *new_filter_f32, new_right_shape,
          storage_type);
      operands.emplace_back(new_filter);

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
      auto matmulOpName = loc_name + "_matmul_" + std::to_string(id);
      LoweringConfig::split_map[loc_name].insert(matmulOpName);
      if (std::distance(value->user_begin(), value->user_end()) == 1 &&
          isa<top::ReshapeOp, top::SqueezeOp>(*(value->user_begin()))) {
        // trick or temp workaround: op order influence layer group
        auto new_reshape_shape =
            module::getShape((*(value->user_begin()))->getResult(0));
        auto elementType =
            module::getElementType((*(value->user_begin()))->getResult(0));
        auto idx = id;
        auto reshapeOp = rewriter.create<top::ReshapeOp>(
            NameLoc::get(rewriter.getStringAttr(loc_name + "_reshape_" +
                                                std::to_string(idx))),
            RankedTensorType::get(new_reshape_shape, elementType),
            ValueRange{matmulOp});
        LoweringConfig::split_map[loc_name].insert(loc_name + "_reshape_" +
                                                   std::to_string(idx));
        id++;
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
          LoweringConfig::split_map[loc_name].insert(loc_name + "_reshape_" +
                                                     std::to_string(id));
          rewriter.replaceOp(value, reshapeOp);
        } else {
          LoweringConfig::split_map[loc_name].insert(matmulOpName);
          rewriter.replaceOp(value, matmulOp);
        }
      }
    }

    return success();
  }
};

/*
Matmul-->Reshape-->Softmax-->Reshape-->Transpose-->Reshape -->\
                                                               Mul-->ReduceSum
==>
                         ***-->Unsqueeze-->Concat-->Reshape-->/
Matmul(slice weight)-->Softmax-->Reshape-->Concat-->Slice-->Reshape-->\
                                                                        Mul-->Add-->ReduceSum
                                                                ***-->/
1)concat(8x32x76760x1x4->8x32x76760x5x4) to slice(8x76760x5x4->8x76760x1x4) to
reduce data move 2)eleminate permute
*/
class ConcatWithReduceSum2SliceWithAdd
    : public OpRewriterPatternEx<top::ReduceOp> {
public:
  ConcatWithReduceSum2SliceWithAdd(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ReduceOp>(
            context, "ConcatWithReduceSum2SliceWithAdd", benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::ReduceOp op,
                      mlir::PatternRewriter &rewriter) const override {
    // TODO: support quantized type
    auto input = op.getInput();
    auto mode = op.getMode();
    if (mode != "ReduceSum")
      return failure();
    auto mul_op = dyn_cast<top::MulOp>(input.getDefiningOp());
    if (!mul_op)
      return failure();
    auto mul_left = mul_op.getInputs()[1];
    auto mul_right = mul_op.getInputs()[0];
    auto mul_left_op = dyn_cast<top::ReshapeOp>(mul_left.getDefiningOp());
    auto mul_right_op = dyn_cast<top::ReshapeOp>(mul_right.getDefiningOp());
    if (!mul_left_op || !mul_right_op)
      return failure();
    auto permute_op =
        dyn_cast<top::PermuteOp>(mul_left_op.getInput().getDefiningOp());
    if (!permute_op)
      return failure();
    auto reshape1_op =
        dyn_cast<top::ReshapeOp>(permute_op.getInput().getDefiningOp());
    if (!reshape1_op)
      return failure();
    auto softmax_op =
        dyn_cast<top::SoftmaxOp>(reshape1_op.getInput().getDefiningOp());
    if (!softmax_op)
      return failure();
    auto reshape2_op =
        dyn_cast<top::ReshapeOp>(softmax_op.getInput().getDefiningOp());
    if (!reshape2_op)
      return failure();
    auto softmax_op_shape = module::getShape(softmax_op->getResult(0));
    int matmul_slice_sec = softmax_op_shape[softmax_op_shape.size() - 2];
    auto matmul_op =
        dyn_cast<top::MatMulOp>(reshape2_op.getInput().getDefiningOp());
    if (!matmul_op)
      return failure();

    auto concat_op =
        dyn_cast<top::ConcatOp>(mul_right_op.getInput().getDefiningOp());
    if (!concat_op)
      return failure();
    int concat_input_num = concat_op.getInputs().size();

    auto matmul_left = matmul_op.getOperand(0);
    auto matmul_right = matmul_op.getOperand(1);
    if (!module::isWeight(matmul_right))
      return failure();
    auto weight_op = matmul_right.getDefiningOp<top::WeightOp>();
    if (!weight_op->hasOneUse())
      return failure();

    std::vector<Value> right_concat_operands;
    for (auto input : concat_op.getInputs()) {
      auto unsqueeze_op = dyn_cast<top::UnsqueezeOp>(input.getDefiningOp());
      if (!unsqueeze_op)
        return failure();
      right_concat_operands.emplace_back(unsqueeze_op.getInput());
    }
    // auto l_trans = matmul_op.getLeftTranspose();
    // auto r_trans = matmul_op.getRightTranspose();
    auto weight_type = module::getElementType(weight_op.getOutput());
    auto weight_shape = module::getShape(weight_op.getOutput());
    auto weight_f32 = weight_op.read<float>();
    std::vector<int64_t> new_weight_shape = weight_shape;
    new_weight_shape[new_weight_shape.size() - 1] /= matmul_slice_sec;

    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    std::vector<Value> concat_operands;
    int32_t new_weight_size =
        std::accumulate(new_weight_shape.begin(), new_weight_shape.end(), 1,
                        std::multiplies<int32_t>());
    int32_t outer_weight_num =
        std::accumulate(new_weight_shape.begin(), new_weight_shape.end() - 1, 1,
                        std::multiplies<int32_t>());
    int32_t weight_last_dim_size = weight_shape[weight_shape.size() - 1];
    int32_t inner_weight_num = new_weight_shape[weight_shape.size() - 1];

    Value bias = matmul_op.getBias();
    bool with_bias = !module::isNone(bias);
    auto bias_shape = module::getShape(bias);
    std::vector<int64_t> new_bias_shape = bias_shape;
    new_bias_shape[new_bias_shape.size() - 1] /= matmul_slice_sec;
    int32_t new_bias_size =
        std::accumulate(new_bias_shape.begin(), new_bias_shape.end(), 1,
                        std::multiplies<int32_t>());
    int32_t outer_bias_num =
        std::accumulate(new_bias_shape.begin(), new_bias_shape.end() - 1, 1,
                        std::multiplies<int32_t>());
    int32_t bias_last_dim_size = bias_shape[bias_shape.size() - 1];
    int32_t inner_bias_num = new_bias_shape[bias_shape.size() - 1];
    std::vector<int64_t> new_matmul_shape =
        module::getShape(matmul_op.getOutput());
    new_matmul_shape[new_matmul_shape.size() - 1] /= matmul_slice_sec;
    auto none_op = module::getNoneOp(op);

    // do 8(matmul) -> 8(softmax) -> 8(reshape) instead of 8(matmul -> softmax
    // ->reshape)
    operands.clear();
    for (int matmul_slice_idx = 0; matmul_slice_idx < matmul_slice_sec;
         matmul_slice_idx++) {
      // Matmul[1x76760x256,256x160,1x1x160] -> 8 *
      // matmul[1x76760x256,256x20,1x1x20] [256x160] -> 8[256x20]
      auto new_weight_f32 =
          std::make_shared<std::vector<float>>(new_weight_size);
      for (int32_t outer_weight_idx = 0; outer_weight_idx < outer_weight_num;
           outer_weight_idx++)
        for (int32_t inner_weight_idx = 0; inner_weight_idx < inner_weight_num;
             inner_weight_idx++)
          new_weight_f32->at(outer_weight_idx * inner_weight_num +
                             inner_weight_idx) =
              weight_f32->at(outer_weight_idx * weight_last_dim_size +
                             matmul_slice_idx * inner_weight_num +
                             inner_weight_idx);
      auto new_weight_type =
          RankedTensorType::get(new_weight_shape, weight_type);
      auto new_filter = top::WeightOp::create<float>(
          matmul_op, "_filter_" + std::to_string(matmul_slice_idx),
          *new_weight_f32, new_weight_type);

      rewriter.setInsertionPointAfter(matmul_op);
      auto new_matmul_op = rewriter.clone(*matmul_op);
      module::setLocSuffix(new_matmul_op, std::to_string(matmul_slice_idx));
      new_matmul_op->setOperand(0, matmul_left);
      new_matmul_op->setOperand(1, new_filter);
      module::setShape(new_matmul_op->getResult(0), new_matmul_shape);

      // [1x1x160] -> 8[1x1x20]
      if (with_bias) {
        auto bias_op = cast<top::WeightOp>(bias.getDefiningOp());
        auto bias_f32 = bias_op.read<float>();
        auto new_bias_f32 = std::make_shared<std::vector<float>>(new_bias_size);
        for (int32_t outer_bias_idx = 0; outer_bias_idx < outer_bias_num;
             outer_bias_idx++)
          for (int32_t inner_bias_idx = 0; inner_bias_idx < inner_bias_num;
               inner_bias_idx++)
            new_bias_f32->at(outer_bias_idx * inner_bias_num + inner_bias_idx) =
                bias_f32->at(outer_bias_idx * bias_last_dim_size +
                             matmul_slice_idx * inner_bias_num +
                             inner_bias_idx);
        auto new_bias_type = RankedTensorType::get(
            new_bias_shape,
            llvm::cast<ShapedType>(bias.getType()).getElementType());
        auto new_bias = top::WeightOp::create(
            matmul_op, "_bias_" + std::to_string(matmul_slice_idx),
            *new_bias_f32, new_bias_type);
        new_matmul_op->setOperand(2, new_bias);
      } else {
        new_matmul_op->setOperand(2, bias);
      }
      operands.emplace_back(new_matmul_op->getResult(0));
    }

    rewriter.setInsertionPointAfterValue(operands[0]);
    for (int matmul_slice_idx = 0; matmul_slice_idx < matmul_slice_sec;
         matmul_slice_idx++) {
      // Softmax[1x76760x8x20,axis=3] -> Softmax[1x76760x20,axis=2]
      attrs.clear();
      attrs.emplace_back(
          rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(2)));
      auto sftmax_name_loc = NameLoc::get(rewriter.getStringAttr(
          module::getName(softmax_op.getOperation()).str() + "_" +
          std::to_string(matmul_slice_idx)));
      auto new_softmax_op = rewriter.create<top::SoftmaxOp>(
          sftmax_name_loc,
          RankedTensorType::get(
              {softmax_op_shape[0], softmax_op_shape[1], softmax_op_shape[3]},
              module::getElementType(softmax_op.getOutput())),
          ValueRange{operands[matmul_slice_idx]}, attrs);
      operands[matmul_slice_idx] = new_softmax_op->getResult(0);
    }

    rewriter.setInsertionPointAfterValue(operands[matmul_slice_sec - 1]);
    for (int matmul_slice_idx = 0; matmul_slice_idx < matmul_slice_sec;
         matmul_slice_idx++) {
      // Reshape[1x76760x20] -> [1x76760x5x4]
      attrs.clear();
      attrs.emplace_back(rewriter.getNamedAttr(
          "shape",
          rewriter.getI64ArrayAttr({softmax_op_shape[0], softmax_op_shape[1],
                                    concat_input_num,
                                    softmax_op_shape[3] / concat_input_num})));
      auto softmax_reshape_op = rewriter.create<top::ReshapeOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(operands[matmul_slice_idx].getDefiningOp())
                  .str() +
              "_reshape")),
          RankedTensorType::get(
              {softmax_op_shape[0], softmax_op_shape[1], concat_input_num,
               softmax_op_shape[3] / concat_input_num},
              module::getElementType(operands[matmul_slice_idx])),
          operands[matmul_slice_idx], attrs);

      concat_operands.emplace_back(softmax_reshape_op->getResult(0));
    }

    // Concat [1x76760x5x4] -> [8x76760x5x4]
    auto reshape_concat_shape = module::getShape(concat_operands[0]);
    attrs.clear();
    rewriter.setInsertionPointAfterValue(concat_operands[matmul_slice_sec - 1]);
    attrs.emplace_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(0)));
    auto concat_type = RankedTensorType::get(
        {reshape_concat_shape[0] * matmul_slice_sec, reshape_concat_shape[1],
         reshape_concat_shape[2], reshape_concat_shape[3]},
        module::getElementType(concat_operands[0]));
    auto new_concat_op = rewriter.create<top::ConcatOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(matmul_op.getOperation()).str() + "_concat")),
        concat_type, concat_operands, attrs);

    std::vector<Value> add_operands;
    operands.clear();
    std::vector<Value> slice_operands;
    for (int slice_idx = 0; slice_idx < concat_input_num; slice_idx++) {
      // Slice [8x76760x5x4] -> [8x76760x1x4]
      attrs.clear();
      slice_operands.clear();
      slice_operands.emplace_back(new_concat_op->getResult(0));
      slice_operands.emplace_back(none_op);
      slice_operands.emplace_back(none_op);
      slice_operands.emplace_back(none_op);
      attrs.emplace_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(2)));
      attrs.emplace_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr({0, 0, slice_idx, 0})));
      attrs.emplace_back(rewriter.getNamedAttr(
          "steps", rewriter.getI64ArrayAttr({1, 1, 1, 1})));
      attrs.emplace_back(rewriter.getNamedAttr(
          "ends",
          rewriter.getI64ArrayAttr({reshape_concat_shape[0] * matmul_slice_sec,
                                    reshape_concat_shape[1], slice_idx + 1,
                                    reshape_concat_shape[3]})));
      auto slice_type = RankedTensorType::get(
          {reshape_concat_shape[0] * matmul_slice_sec, reshape_concat_shape[1],
           1, reshape_concat_shape[3]},
          module::getElementType(new_concat_op->getResult(0)));
      auto slice_op = rewriter.create<top::SliceOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(matmul_op.getOperation()).str() + "_slice_" +
              std::to_string(slice_idx))),
          slice_type, slice_operands, attrs);
      operands.emplace_back(slice_op->getResult(0));
    }
    for (int slice_idx = 0; slice_idx < concat_input_num; slice_idx++) {
      // Reshape [8x76760x1x4] -> [8x1x76760x4]
      attrs.clear();
      attrs.emplace_back(rewriter.getNamedAttr(
          "shape", rewriter.getI64ArrayAttr(
                       {reshape_concat_shape[0] * matmul_slice_sec, 1,
                        reshape_concat_shape[1], reshape_concat_shape[3]})));
      auto slice_reshape_op = rewriter.create<top::ReshapeOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(operands[slice_idx].getDefiningOp()).str() +
              "_reshape")),
          RankedTensorType::get({reshape_concat_shape[0] * matmul_slice_sec, 1,
                                 reshape_concat_shape[1],
                                 reshape_concat_shape[3]},
                                module::getElementType(operands[slice_idx])),
          operands[slice_idx], attrs);
      operands[slice_idx] = slice_reshape_op->getResult(0);
    }
    rewriter.setInsertionPointAfterValue(
        right_concat_operands[concat_input_num - 1]);
    for (int slice_idx = 0; slice_idx < concat_input_num; slice_idx++) {
      // Mul [8x1x76760x4,8x32x76760x4]
      attrs.clear();
      auto new_mul_op = rewriter.clone(*mul_op);
      module::setLocSuffix(new_mul_op, std::to_string(slice_idx));
      new_mul_op->setOperand(0, operands[slice_idx]);
      new_mul_op->setOperand(1, right_concat_operands[slice_idx]);
      std::vector<int64_t> new_mul_shape =
          module::getShape(operands[slice_idx]);
      new_mul_shape[1] =
          std::max(new_mul_shape[1],
                   module::getShape(right_concat_operands[slice_idx])[1]);
      module::setShape(new_mul_op->getResult(0), new_mul_shape);
      add_operands.emplace_back(new_mul_op->getResult(0));
    }

    // Add [8x32xl76760x4,8x32x76760x4]
    // rewriter.setInsertionPointAfterValue(operands[0]);
    auto add_loc = NameLoc::get(
        rewriter.getStringAttr(module::getName(mul_op.getOperation()).str() +
                               "_add_" + std::to_string(0)));
    auto add_op = rewriter.create<top::AddOp>(
        add_loc, add_operands[0].getType(),
        mlir::ValueRange{add_operands[0], add_operands[1]});
    for (int add_idx = 1; add_idx < add_operands.size() - 1; add_idx++) {
      // if(operands[i+1].getDefiningOp()->getLoc() > add_op->getLoc())
      //   insertpoint = operands[1];
      // rewriter.setInsertionPointAfterValue(add_op);
      add_loc = NameLoc::get(
          rewriter.getStringAttr(module::getName(mul_op.getOperation()).str() +
                                 "_add_" + std::to_string(add_idx)));
      add_op = rewriter.create<top::AddOp>(
          add_loc, add_operands[add_idx].getType(),
          mlir::ValueRange{add_op.getOutput(), add_operands[add_idx + 1]});
    }

    // Reduce [8x32x76760x4] -> [8x32x76760]
    attrs.clear();
    operands.clear();
    std::vector<int64_t> reducesum_shape = module::getShape(op.getOutput());
    auto reducesum_type = module::getTypeLike(op.getOutput(), reducesum_shape);
    auto loc_reducesum = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getOperation()).str() + "_new"));
    operands.emplace_back(add_op.getOutput());
    // for (int i = operands.size(); i < 3; i++) {
    //   operands.emplace_back(none_op);
    // }
    attrs.emplace_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr(mode)));
    attrs.emplace_back(rewriter.getNamedAttr("axes", op.getAxes()));
    attrs.emplace_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    auto reduce_op = rewriter.create<top::ReduceOp>(
        loc_reducesum, reducesum_type, operands, attrs);

    rewriter.setInsertionPointAfter(op);
    rewriter.replaceAllUsesWith(op, reduce_op->getResult(0));
    // rewriter.eraseOp(bias_op);
    // rewriter.eraseOp(op);
    // rewriter.eraseOp(weight_op);
    return success();
  }
};

// concat 6([1,1,64,320,320]) + reducesum([6,1,64,320,320], keepdims=false) -> 5
// add [1,1,64,320,320] + reshape concat is inplace_op, but reduce performance
// is low with transpose to move reduce axis to H/W; can be removed if backend
// supports reduce at N/C
class ConcatReduceSum2AddReshape : public OpRewriterPatternEx<top::ReduceOp> {
public:
  ConcatReduceSum2AddReshape(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ReduceOp>(
            context, "ConcatReduceSum2AddReshape", benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::ReduceOp op,
                      mlir::PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto mode = op.getMode();
    auto axes = module::getI64Array(op.getAxes());
    auto input_shape = module::getShape(input);
    // TODO : performance may decline after rewriting ; support other mode
    if (mode != "ReduceSum" || input_shape.size() != 5 || axes->size() != 1 ||
        axes->at(0) != 0)
      return failure();
    auto concat_op = dyn_cast<top::ConcatOp>(op.getInput().getDefiningOp());
    if (!concat_op || !concat_op->hasOneUse())
      return failure();
    int concat_input_num = concat_op.getInputs().size();

    auto add_loc = NameLoc::get(
        rewriter.getStringAttr(module::getName(op.getOperation()).str() +
                               "_add_" + std::to_string(0)));
    auto add_op = rewriter.create<top::AddOp>(
        add_loc, op.getOutput().getType(),
        mlir::ValueRange{concat_op.getInputs()[0], concat_op.getInputs()[1]});
    module::setShape(add_op->getResult(0),
                     module::getShape(concat_op.getInputs()[0]));
    for (int add_idx = 1; add_idx < concat_input_num - 1; add_idx++) {
      // if(operands[i+1].getDefiningOp()->getLoc() > add_op->getLoc())
      //   insertpoint = operands[1];
      // rewriter.setInsertionPointAfterValue(add_op);
      add_loc = NameLoc::get(
          rewriter.getStringAttr(module::getName(op.getOperation()).str() +
                                 "_add_" + std::to_string(add_idx)));
      add_op = rewriter.create<top::AddOp>(
          add_loc, op.getOutput().getType(),
          mlir::ValueRange{add_op.getOutput(),
                           concat_op.getInputs()[add_idx + 1]});
      module::setShape(add_op->getResult(0),
                       module::getShape(concat_op.getInputs()[0]));
    }

    std::vector<NamedAttribute> attrs;
    // attrs.emplace_back(
    //       rewriter.getNamedAttr("const_val",
    //       rewriter.getF64FloatAttr(1/concat_input_num)));
    // auto mulconst_loc = NameLoc::get(
    //     rewriter.getStringAttr(module::getName(op.getOperation()).str() +
    //                              "_mul"));
    // auto mulconst_op = rewriter.create<top::MulConstOp>(
    //     mulconst_loc, op.getOutput().getType(),
    //     mlir::ValueRange{add_op.getOutput()},attrs);
    // module::setShape(mulconst_op->getResult(0),
    // module::getShape(add_op.getOutput()));

    if (!op.getKeepdims()) {
      auto add_shape = module::getShape(add_op->getResult(0));
      attrs.clear();
      attrs.emplace_back(rewriter.getNamedAttr(
          "shape", rewriter.getI64ArrayAttr({add_shape[1], add_shape[2],
                                             add_shape[3], add_shape[4]})));
      auto add_reshape_op = rewriter.create<top::ReshapeOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(add_op.getOutput()).str() + "_reshape")),
          RankedTensorType::get(
              {add_shape[1], add_shape[2], add_shape[3], add_shape[4]},
              module::getElementType(add_op.getOutput())),
          add_op.getOutput(), attrs);
      rewriter.replaceAllUsesWith(op, add_reshape_op->getResult(0));
      return success();
    }

    rewriter.replaceAllUsesWith(op, add_op->getResult(0));
    return success();
  }
};

// x / sqrt(y) => x * rsqrt(y)
class ConvertToRSqrt : public OpRewriterPatternEx<top::DivOp> {
public:
  ConvertToRSqrt(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::DivOp>(context, "ConvertToRSqrt", benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::DivOp op,
                      mlir::PatternRewriter &rewriter) const override {
    auto divisor = op.getIsReverse() ? op->getOperand(0) : op->getOperand(1);
    auto dividend = op.getIsReverse() ? op->getOperand(1) : op->getOperand(0);
    auto def_op = divisor.getDefiningOp();
    if (!def_op->hasOneUse())
      return failure();
    if (auto sqrtOp = dyn_cast<top::SqrtOp>(def_op)) {
      auto in = def_op->getOperand(0);
      // rsqrt
      auto rsqrtOp = rewriter.create<top::RsqrtOp>(
          NameLoc::get(
              rewriter.getStringAttr(module::getName(divisor).str() + "_inv")),
          divisor.getType(), ValueRange{in});
      auto out = op.getOutput();
      // mul
      auto mulOp = rewriter.create<top::MulOp>(
          out.getLoc(), out.getType(),
          ValueRange{dividend, rsqrtOp.getOutput()}, op->getAttrs());
      mulOp->removeAttr(rewriter.getStringAttr("is_reverse"));
      out.replaceAllUsesWith(mulOp.getOutput());
    } else {
      return failure();
    }
    return success();
  }
};

// abs(x) * abs(x) => x * x
class ConvertToSquare : public OpRewriterPatternEx<top::MulOp> {
public:
  ConvertToSquare(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MulOp>(context, "ConvertToSquare", benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MulOp op,
                      mlir::PatternRewriter &rewriter) const override {
    auto in0 = op->getOperand(0);
    auto in1 = op->getOperand(1);
    if (in0 != in1)
      return failure();
    auto def_op = in0.getDefiningOp();
    if (!def_op->hasOneUse())
      return failure();
    if (auto absOp = dyn_cast<top::AbsOp>(def_op)) {
      auto in = def_op->getOperand(0);
      op->setOperands({in, in});
    } else {
      return failure();
    }
    return success();
  }
};

// x * sigmoid(1.7020000219345093 * x) => QGELU(x)
class ConvertToQGELU : public OpRewriterPatternEx<top::MulOp> {
public:
  ConvertToQGELU(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::MulOp>(context, "ConvertToQGELU", benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::MulOp op,
                      mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2)
      return failure();
    if (op.getDoRelu())
      return failure();
    auto ins = op->getOperands();
    Operation *def_ops[2] = {nullptr, nullptr};
    bool is_sigmoid[2] = {false, false};
    for (auto i = 0; i < 2; ++i) {
      def_ops[i] = ins[i].getDefiningOp();
      is_sigmoid[i] =
          isa<top::SigmoidOp>(def_ops[i]) && def_ops[i]->hasOneUse();
    }
    if (!is_sigmoid[0] && !is_sigmoid[1])
      return failure();
    int j = -1;
    for (auto i = 0; i < 2; ++i) {
      auto x = ins[1 - i];
      if (is_sigmoid[i]) {
        auto sigmOp = dyn_cast<top::SigmoidOp>(def_ops[i]);
        auto sigm_in = sigmOp.getInput();
        auto may_be_mulc_op = sigm_in.getDefiningOp();
        if (!may_be_mulc_op->hasOneUse())
          continue;
        if (auto mulcOp = dyn_cast<top::MulConstOp>(may_be_mulc_op)) {
          if (mulcOp.getDoRelu())
            continue;
          if (mulcOp.getConstVal().convertToDouble() != 1.7020000219345093)
            continue;
          if (x != mulcOp.getInput())
            continue;
          j = 1 - i;
        }
      }
    }
    if (j == -1)
      return failure();
    auto out = op.getOutput();
    llvm::SmallVector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("approx_mode", rewriter.getStringAttr("sigm")));
    auto qgeluOp = rewriter.create<top::GELUOp>(out.getLoc(), out.getType(),
                                                ValueRange{ins[j]}, attrs);
    out.replaceAllUsesWith(qgeluOp.getOutput());
    return success();
  }
};

// interp (nearst_mode) using gather -> matmul
// gather(8x64x32x32,64,axis=2) + gather(8x64x64x32,64,axis=3)->
// matmul(8x64x32x32,1x1x32x64) + matmul(1x1x64x32,8x64x32x64)

class InterpNearst2Matmul : public OpRewriterPatternEx<top::GatherOp> {
public:
  InterpNearst2Matmul(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::GatherOp>(context, "InterpNearst2Matmul") {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::GatherOp op,
                      mlir::PatternRewriter &rewriter) const override {

    auto gather_input = op.getOperand(0);
    auto gather_op = dyn_cast<top::GatherOp>(gather_input.getDefiningOp());
    if (!gather_op) {
      return failure();
    }

    auto gather_shape = module::getShape(gather_op.getInput());
    auto op_weight =
        dyn_cast<top::WeightOp>((op.getOperands()[1]).getDefiningOp());
    if (!op_weight) {
      return failure();
    }
    auto gather_weight =
        dyn_cast<top::WeightOp>((gather_op.getOperands()[1]).getDefiningOp());
    if (!gather_weight) {
      return failure();
    }
    std::vector<int> unsample_ratio = {0, 0};
    if (op.getAxis() == gather_shape.size() - 1 ||
        op.getAxis() == gather_shape.size() - 2) {
      checkWeightUpsample<float>(op_weight, gather_shape, unsample_ratio,
                                 op.getAxis());
    }
    if (gather_op.getAxis() == gather_shape.size() - 1 ||
        gather_op.getAxis() == gather_shape.size() - 2) {
      checkWeightUpsample<float>(gather_weight, gather_shape, unsample_ratio,
                                 gather_op.getAxis());
    }
    // not nearst_mode or shape don't change
    if (unsample_ratio[0] * unsample_ratio[1] <= 1)
      return failure();

    rewriter.setInsertionPointAfter(op);
    std::string weightName =
        module::getName(gather_op.getOperands()[0]).str() + "_weight";
    auto weightType = RankedTensorType::get(
        {1, 1, gather_shape[2], gather_shape[3] * unsample_ratio[1]},
        module::getElementType(gather_op.getOperands()[0]));
    auto weight_size = weightType.getNumElements();
    auto weightCoeff = std::make_shared<std::vector<float>>(weight_size, 0);

    for (int row_idx = 0; row_idx < gather_shape[2]; row_idx++) {
      weightCoeff->at(row_idx * gather_shape[3] * unsample_ratio[1] +
                      row_idx * unsample_ratio[1]) = 1;
      weightCoeff->at(row_idx * gather_shape[3] * unsample_ratio[1] +
                      row_idx * unsample_ratio[1] + 1) = 1;
    }
    auto wret = module::weightFile().addTensor(
        weightName, (float *)weightCoeff->data(), weightType);
    assert(succeeded(wret));
    auto weight_op0 = rewriter.create<top::WeightOp>(
        NameLoc::get(rewriter.getStringAttr(weightName)), weightType,
        ValueRange{});

    auto none = module::getNoneOp(op);
    auto matmul0 = rewriter.create<top::MatMulOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(gather_op.getOperation()).str() + "_2matmul")),
        RankedTensorType::get({gather_shape[0], gather_shape[1],
                               gather_shape[2],
                               gather_shape[3] * unsample_ratio[1]},
                              module::getElementType(gather_op.getOutput())),
        ValueRange{gather_op.getInput(), weight_op0, none}, op->getAttrs());

    weightName = module::getName(op.getOperands()[0]).str() + "_weight";
    weightType = RankedTensorType::get(
        {1, 1, gather_shape[2] * unsample_ratio[0], gather_shape[3]},
        module::getElementType(op.getOperands()[0]));
    weight_size = weightType.getNumElements();
    weightCoeff = std::make_shared<std::vector<float>>(weight_size, 0);

    for (int row_idx = 0; row_idx < gather_shape[2] * unsample_ratio[0];
         row_idx++) {
      weightCoeff->at(row_idx * gather_shape[3] + row_idx / unsample_ratio[0]) =
          1;
    }
    wret = module::weightFile().addTensor(
        weightName, (float *)weightCoeff->data(), weightType);
    assert(succeeded(wret));
    auto weight_op1 = rewriter.create<top::WeightOp>(
        NameLoc::get(rewriter.getStringAttr(weightName)), weightType,
        ValueRange{});

    auto matmul1 = rewriter.create<top::MatMulOp>(
        op.getLoc(),
        RankedTensorType::get({gather_shape[0], gather_shape[1],
                               gather_shape[2] * unsample_ratio[0],
                               gather_shape[3] * unsample_ratio[1]},
                              module::getElementType(op.getOutput())),
        ValueRange{weight_op1, matmul0.getOutput(), none}, op->getAttrs());

    op.replaceAllUsesWith(matmul1.getOutput());
    rewriter.eraseOp(op);
    rewriter.eraseOp(gather_op);
    return success();
  }

private:
  template <typename T>
  void checkWeightUpsample(top::WeightOp op,
                           llvm::ArrayRef<int64_t> origin_shape,
                           std::vector<int> &unsample_ratio, int axis) const {
    auto weight = op.read<T>();
    auto weight_size = weight->size();
    int origin_size = origin_shape[axis];
    if (weight_size % origin_size != 0) {
      unsample_ratio[axis - (origin_shape.size() - 2)] = 0;
      return;
    }
    int ratio = weight_size / origin_size;
    for (int idx = 0; idx < weight_size; idx++) {
      if ((int32_t)weight->at(idx) != idx / ratio) {
        unsample_ratio[axis - (origin_shape.size() - 2)] = 0;
        return;
      }
    }
    unsample_ratio[axis - (origin_shape.size() - 2)] = ratio;
    return;
  }
};

// permute -> reshape -> avgpool -> gridsampler1
//                    -> gridsampler2
//                    =>
// reshape -> avgpool -> permute -> gridsampler1
//         -> permute -> gridsampler2

struct PermuteBeforeGridSampler : public OpRewriterPatternEx<top::PermuteOp> {
public:
  PermuteBeforeGridSampler(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::PermuteOp>(context, "PermuteBeforeGridSampler",
                                            benefit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::PermuteOp op,
                                    PatternRewriter &rewriter) const override {

    if (!op.getOutput().hasOneUse() ||
        module::getShape(op.getInput()).size() != 5)
      return failure();

    auto next_op = *op.getOutput().user_begin();
    auto reshape = dyn_cast<top::ReshapeOp>(next_op);
    if (!reshape)
      return failure();

    SmallVector<Operation *> grid_samplers;
    auto reshape_users = reshape->getResult(0).getUsers();
    for (auto user : reshape_users) {
      if (isa<top::AvgPoolOp>(user)) {
        for (auto avgpool_user : user->getResult(0).getUsers()) {
          if (auto grid_sampler = dyn_cast<top::GridSamplerOp>(avgpool_user)) {
            std::vector<int64_t> grid_sampler_shape =
                module::getShape(grid_sampler);
            if (grid_sampler_shape.size() != 4 || grid_sampler_shape[2] != 1)
              return failure();
            grid_samplers.push_back(grid_sampler);
          } else {
            return failure();
          }
        }
      } else if (auto grid_sampler = dyn_cast<top::GridSamplerOp>(user)) {
        std::vector<int64_t> grid_sampler_shape =
            module::getShape(grid_sampler);
        if (grid_sampler_shape.size() != 4 || grid_sampler_shape[2] != 1)
          return failure();
        grid_samplers.push_back(grid_sampler);
      } else {
        return failure();
      }
    }

    rewriter.updateRootInPlace(reshape, [&]() {
      auto reshape_loc = module::getName(reshape.getOutput()).str();
      auto in_shape = module::getShape(op.getInput());
      std::vector<int64_t> reshape_shape{in_shape[0], in_shape[1], in_shape[2],
                                         in_shape[3] * in_shape[4]};
      auto reshape_type = RankedTensorType::get(
          reshape_shape, module::getElementType(op.getInput()));
      reshape->setOperand(0, op.getInput());
      reshape->setLoc(
          NameLoc::get(rewriter.getStringAttr(reshape_loc + "_permute")));
      reshape.getResult().setType(reshape_type);
      reshape->setAttr("shape", rewriter.getI64ArrayAttr(reshape_shape));
    });

    for (auto user : reshape_users) {
      if (auto avg_pool = dyn_cast<top::AvgPoolOp>(user)) {
        auto input_type =
            avg_pool.getInput().getType().cast<RankedTensorType>();
        auto input_shape = input_type.getShape();

        SmallVector<int64_t> old_kernel;
        for (auto dim : avg_pool.getKernelShape().getValue()) {
          old_kernel.push_back(dim.cast<IntegerAttr>().getInt());
        }

        SmallVector<int64_t> kernel = {old_kernel[1], old_kernel[0]};
        SmallVector<int64_t> strides = kernel;

        int64_t in_height = input_shape[2];
        int64_t kernel_height = kernel[0];
        int64_t strides_height = strides[0];
        int64_t out_height = (in_height - kernel_height) / strides_height + 1;

        SmallVector<int64_t> output_shape{input_shape[0], input_shape[1],
                                          out_height, input_shape[3]};

        auto output_type =
            RankedTensorType::get(output_shape, input_type.getElementType());
        auto avg_pool_loc = module::getName(avg_pool.getOutput()).str();
        rewriter.updateRootInPlace(avg_pool, [&]() {
          // auto avg_pool_type =
          //   reshape.getOutput().getType().cast<RankedTensorType>();
          avg_pool->setLoc(
              NameLoc::get(rewriter.getStringAttr(avg_pool_loc + "_hwtrans")));
          avg_pool->setAttr("kernel_shape", rewriter.getI64ArrayAttr(kernel));
          avg_pool->setAttr("strides", rewriter.getI64ArrayAttr(strides));
          avg_pool.getResult().setType(output_type);
        });
      }
    }
    std::vector<int64_t> permute_order{3, 1, 0, 2};
    for (auto user : grid_samplers) {
      auto grid_sampler = dyn_cast<top::GridSamplerOp>(user);
      rewriter.setInsertionPoint(grid_sampler);
      auto grid_sampler_loc = module::getName(grid_sampler.getOutput()).str();

      auto input_value = grid_sampler.getOperand(0);
      auto input_type = input_value.getType().cast<RankedTensorType>();
      auto input_shape = input_type.getShape();
      SmallVector<int64_t> output_shape;
      for (auto dim : permute_order) {
        output_shape.push_back(input_shape[dim]);
      }
      auto output_type =
          RankedTensorType::get(output_shape, input_type.getElementType());
      auto permute_op = rewriter.create<top::PermuteOp>(
          NameLoc::get(rewriter.getStringAttr(grid_sampler_loc + "_permute")),
          output_type, input_value,
          rewriter.getNamedAttr("order",
                                rewriter.getI64ArrayAttr(permute_order)));
      grid_sampler->setOperand(0, permute_op.getOutput());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// gridsampler1 -> reshape -> concat -> permute
// girdsampler2 -> reshape
//                        =>
// gridsampler1 -> permute -> reshape -> concat
// girdsampler2 -> permute -> reshape

class PermuteAfterGridSampler : public OpRewriterPatternEx<top::PermuteOp> {
public:
  PermuteAfterGridSampler(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::PermuteOp>(context, "PermuteAfterGridSampler",
                                            benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::PermuteOp op,
                      mlir::PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (!in_op || !isa<top::ConcatOp>(in_op)) {
      return failure();
    }
    auto concat_op = cast<top::ConcatOp>(in_op);
    if (concat_op.getInputs().size() != 2) {
      return failure();
    }

    SmallVector<top::GridSamplerOp> gridsample_ops;
    SmallVector<top::ReshapeOp> reshape_ops;
    SmallVector<Type> reshape_output_types;

    for (auto input : concat_op.getInputs()) {
      auto def_op = input.getDefiningOp();
      if (!def_op || !isa<top::ReshapeOp>(def_op)) {
        return failure();
      }
      auto reshape_op = cast<top::ReshapeOp>(def_op);
      reshape_ops.push_back(reshape_op);
      reshape_output_types.push_back(reshape_op.getOutput().getType());

      auto reshape_input = reshape_op.getInput().getDefiningOp();
      if (!reshape_input || !isa<top::GridSamplerOp>(reshape_input)) {
        return failure();
      }
      auto grid_sampler = dyn_cast<top::GridSamplerOp>(reshape_input);
      std::vector<int64_t> grid_sampler_shape = module::getShape(grid_sampler);
      if (grid_sampler_shape.size() != 4 || grid_sampler_shape[2] != 1)
        return failure();
      gridsample_ops.push_back(grid_sampler);
    }

    rewriter.startRootUpdate(op);
    auto out_shape = module::getShape(op.getOutput());

    auto in_shape_1 = module::getShape(gridsample_ops[0].getOutput());
    auto in_shape_2 = module::getShape(gridsample_ops[1].getOutput());
    if (in_shape_1 != in_shape_2) {
      return failure();
    }

    auto permute_loc = module::getName(op.getOutput()).str();
    std::vector<int64_t> permute_order{2, 1, 3, 0};
    auto grid_output_type =
        gridsample_ops[0].getOutput().getType().cast<RankedTensorType>();
    auto grid_shape = grid_output_type.getShape();
    SmallVector<int64_t> permuted_shape = {
        grid_shape[permute_order[0]],
        grid_shape[permute_order[1]],
        grid_shape[permute_order[2]],
        grid_shape[permute_order[3]],
    };
    auto permuted_type = RankedTensorType::get(
        permuted_shape, grid_output_type.getElementType());

    SmallVector<int64_t> new_reshape_shape = {
        1, permuted_shape[0] * permuted_shape[1] * permuted_shape[2],
        out_shape[2], out_shape[3]};
    auto reshape_output_type =
        reshape_ops[0].getOutput().getType().cast<RankedTensorType>();
    auto new_reshape_type = RankedTensorType::get(
        new_reshape_shape, reshape_output_type.getElementType());
    for (int i = 0; i < 2; ++i) {
      rewriter.setInsertionPointAfter(gridsample_ops[i]);
      auto new_permute = rewriter.create<top::PermuteOp>(
          NameLoc::get(rewriter.getStringAttr(permute_loc + "_Reshape_" +
                                              std::to_string(i))),
          permuted_type, gridsample_ops[i].getOutput(),
          rewriter.getNamedAttr("order",
                                rewriter.getI64ArrayAttr(permute_order)));
      auto reshape_loc = module::getName(reshape_ops[i].getOutput()).str();
      ;
      reshape_ops[i]->setLoc(
          NameLoc::get(rewriter.getStringAttr(reshape_loc + "_Concat")));
      reshape_ops[i].getInputMutable().assign(new_permute.getOutput());
      reshape_ops[i].setShapeAttr(rewriter.getI64ArrayAttr(new_reshape_shape));
      reshape_ops[i].getResult().setType(new_reshape_type);
    }
    auto concat_loc = module::getName(op.getOutput()).str();
    concat_op->setLoc(NameLoc::get(rewriter.getStringAttr(concat_loc)));
    concat_op.setAxisAttr(rewriter.getSI32IntegerAttr(1));
    concat_op.getResult().setType(op.getOutput().getType());
    rewriter.replaceOp(op, concat_op.getOutput());
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

class ReshapeBeforeGridSampler : public OpRewriterPatternEx<top::ReshapeOp> {
public:
  ReshapeBeforeGridSampler(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ReshapeOp>(context, "ReshapeBeforeGridSampler",
                                            benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::ReshapeOp op,
                      mlir::PatternRewriter &rewriter) const override {

    for (auto next_op : op->getUsers()) {
      Operation *add_op = nullptr;
      if (isa<top::AddOp>(next_op)) {
        add_op = next_op;
      } else if (isa<top::MulConstOp>(next_op)) {
        auto users = next_op->getUsers();
        if (users.empty())
          return failure();
        add_op = *users.begin();
      } else {
        return failure();
      }

      auto add_user = add_op->getUsers();
      if (add_user.empty())
        return failure();
      auto scale_op = *add_user.begin();
      if (!isa<top::MulConstOp>(scale_op))
        return failure();

      auto mulconst_user = scale_op->getUsers();
      if (mulconst_user.empty())
        return failure();
      auto mean_op = *mulconst_user.begin();
      if (!isa<top::AddConstOp>(mean_op))
        return failure();

      auto addconst_user = mean_op->getUsers();
      if (addconst_user.empty())
        return failure();
      auto concat_op = *addconst_user.begin();
      if (!isa<top::ConcatOp>(concat_op))
        return failure();

      auto concat_user = concat_op->getUsers();
      if (concat_user.empty())
        return failure();
      auto grid_op = *concat_user.begin();
      if (!isa<top::GridSamplerOp>(grid_op))
        return failure();
    }

    std::vector<int64_t> reshape_shape = module::getShape(op);
    if (reshape_shape.size() != 4 || reshape_shape[2] != 1)
      return failure();

    auto eleType = module::getElementType(op.getInput());
    auto reshape_loc = module::getName(op.getOutput()).str();
    std::vector<int64_t> new_shape = {reshape_shape[3], reshape_shape[1],
                                      reshape_shape[2], reshape_shape[0]};
    auto new_reshape = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(reshape_loc + "_nwtrans")),
        RankedTensorType::get(new_shape, eleType), op.getInput(),
        rewriter.getNamedAttr("shape", rewriter.getI64ArrayAttr(new_shape)));

    auto new_reshape_out = new_reshape.getOutput();
    rewriter.replaceOp(op, new_reshape_out);
    SmallVector<top::MulConstOp, 4> mulconsts;
    SmallVector<top::AddOp, 4> adds;
    for (Operation *user : new_reshape_out.getUsers()) {
      if (auto mul = dyn_cast<top::MulConstOp>(user)) {
        mulconsts.push_back(mul);
        auto add = dyn_cast<top::AddOp>(*mul->getResult(0).getUsers().begin());
        adds.push_back(add);
      } else if (auto add = dyn_cast<top::AddOp>(user)) {
        adds.push_back(add);
      }
    }
    for (auto old_mulconst : mulconsts) {
      auto inTy = new_reshape_out.getType().cast<RankedTensorType>();
      Type eltTy = inTy.getElementType();
      SmallVector<int64_t, 4> new_shape({inTy.getDimSize(0), inTy.getDimSize(1),
                                         inTy.getDimSize(2),
                                         inTy.getDimSize(3)});
      rewriter.setInsertionPoint(old_mulconst);
      auto mulconst_loc = module::getName(old_mulconst.getOutput()).str();
      auto const_val = old_mulconst.getConstVal().convertToDouble();
      auto new_mulconst = rewriter.create<top::MulConstOp>(
          NameLoc::get(rewriter.getStringAttr(mulconst_loc + "_nwtrans")),
          RankedTensorType::get(new_shape, eltTy), new_reshape_out,
          rewriter.getNamedAttr("const_val",
                                rewriter.getF64FloatAttr(const_val)));
      old_mulconst.replaceAllUsesWith(new_mulconst.getResult());
      rewriter.eraseOp(old_mulconst);
      new_reshape_out = new_mulconst.getResult();
    }

    std::vector<int64_t> permute_order = {3, 1, 2, 0};
    for (auto add : adds) {
      auto oldTy = add.getResult().getType().cast<RankedTensorType>();
      Type eltTy = oldTy.getElementType();

      SmallVector<int64_t, 4> outshape;
      for (int idx : permute_order)
        outshape.push_back(oldTy.getDimSize(idx));
      auto addResTy = RankedTensorType::get(outshape, eltTy);
      auto add_loc = module::getName(add.getOutput()).str();
      add.getResult().setType(addResTy);
      add->setLoc(NameLoc::get(rewriter.getStringAttr(add_loc + "_nwtrans")));

      rewriter.setInsertionPointAfter(add);
      auto perm = rewriter.create<top::PermuteOp>(
          NameLoc::get(rewriter.getStringAttr(add_loc)), oldTy, add.getResult(),
          rewriter.getNamedAttr("order",
                                rewriter.getI64ArrayAttr(permute_order)));
      add.getResult().replaceAllUsesExcept(perm.getResult(), perm);
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
      ->add<ChatGLM3ToGQAAttention, ConvertMatMulWithRightTranspose,
            ConvertMatMul2Attention, ReshapeReorderPattern,
            ConvertMultiInputAdd, WhereBroadcastToTile, ConvertConv2DToImg2Col,
            SplitMatMulPattern, ConvertScaleOp, ConcatToSwapDimInner,
            //  ConcatWithReduceSum2SliceWithAdd,
            ConcatReduceSum2AddReshape, ConvertToRSqrt, ConvertToSquare,
            ConvertToQGELU, InterpNearst2Matmul, ForwardRequantInt,
            PermuteBeforeGridSampler, PermuteAfterGridSampler,
            ReshapeBeforeGridSampler>(patterns->getContext(), 8);
}
} // namespace top
} // namespace tpu_mlir
