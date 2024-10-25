//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

struct TopScaleDotProductAttentionSplitV2 : public OpRewriterPatternEx<ScaleDotProductAttentionOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopScaleDotProductAttentionSplitV2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ScaleDotProductAttentionOp>(context, "TopScaleDotProductAttentionSplitV2") {}

  LogicalResult matchAndRewriteImpl(ScaleDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override {
    // matmul , div, softmax, matmul
    auto query_op = op.getQuery();
    auto key_op   = op.getKey();
    auto value_op = op.getValue();
    auto mask_op  = op.getMask();
    auto scaled   = op.getScale().convertToDouble();
    auto none = module::getNoneOp(op);
    if(!module::isNone(mask_op)) {
      UNREACHABLE_OP("Not support ScaleDotProductAttention when mask is not none.",op);
    }
    std::string query_name = module::getName(query_op).str();
    std::string key_name   = module::getName(key_op).str();
    std::string value_name = module::getName(value_op).str();
    auto query_shape = module::getShape(query_op);
    auto key_shape   = module::getShape(key_op);
    auto value_shape = module::getShape(value_op);
    auto shape_len   = query_shape.size();
    if (shape_len != 4) return failure();
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    ArrayRef<int64_t> order = {0,1,3,2};
    rewriter.setInsertionPointAfter(op);
    RankedTensorType newType, outputType;
    NameLoc loc;
    // k permute
    loc = NameLoc::get(rewriter.getStringAttr(key_name + "_permute"));
    auto permute_type = RankedTensorType::get({key_shape[0], key_shape[1], key_shape[3], key_shape[2]}, module::getElementType(key_op));
    attrs.emplace_back(rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    auto key_permute = rewriter.create<PermuteOp>(loc, permute_type, ValueRange({key_op}), attrs);
    attrs.clear();
    newType = RankedTensorType::get({query_shape[0], query_shape[1], query_shape[2],key_shape[2]}, module::getElementType(query_op));
    outputType = RankedTensorType::get({query_shape[0], query_shape[1], query_shape[2], value_shape[3]}, module::getElementType(value_op));
    loc = NameLoc::get(rewriter.getStringAttr(query_name + "_kvmatmul"));
    operands.push_back(query_op);
    operands.push_back(key_permute);
    operands.push_back(none);
    auto matmulKV = rewriter.create<MatMulOp>(loc, newType, operands, attrs);
    operands.clear();
    // kv*scaled
    loc = NameLoc::get(rewriter.getStringAttr(query_name + "_kvmatmul_with_scaled"));
    if(scaled == 0.0f){
      scaled = (1/sqrt(query_shape[shape_len - 1]));
    }
    attrs.emplace_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(scaled)));
    auto kv_scaled = rewriter.create<MulConstOp>(loc, newType, ValueRange({matmulKV}), attrs);
    attrs.clear();
    // softmax(kv_scaled)
    loc = NameLoc::get(rewriter.getStringAttr(query_name + "_kv_scaled_softmax"));
    attrs.emplace_back(rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(shape_len - 1)));
    auto softmax = rewriter.create<SoftmaxOp>(loc, newType, ValueRange({kv_scaled}), attrs);
    attrs.clear();
    // softmax(kv_scaled)*value
    loc = NameLoc::get(rewriter.getStringAttr(query_name + "_kv_scaled_softmax_mul_value"));
    operands.push_back(softmax);
    operands.push_back(value_op);
    operands.push_back(none);
    auto kv_scaled_softmax_mul_value = rewriter.create<MatMulOp>(loc, outputType, operands, attrs);
    rewriter.replaceAllUsesWith(op, kv_scaled_softmax_mul_value);
    rewriter.eraseOp(op);
    return success();
  }
};


void ScaleDotProductAttentionOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<TopScaleDotProductAttentionSplitV2>(context);
}
