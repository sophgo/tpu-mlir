//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace mlir;
using namespace tpu_mlir::top;

Value get_weight(Value weight, int head, int idx, int axis, Type to_type, std::string base_name) {
  auto op = weight.getDefiningOp();
  if (module::isWeight(weight)) {
    auto shape = module::getShape(weight);
    auto dim = shape.size();
    axis = axis < 0 ? dim + axis : axis;
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) {
      outer *= shape[i];
    }
    int64_t inner = module::getNumElements(weight) / outer;
    int64_t head_inner = inner / head;
    auto out_weight = std::make_shared<std::vector<float_t>>(outer * head_inner);
    auto weight_op = cast<top::WeightOp>(weight.getDefiningOp()).read_as_float();
    for (int64_t i = 0; i < outer; ++i) {
      int64_t src_offset = i * inner + idx * head_inner;
      int64_t dst_offset = i * head_inner;
      for (int64_t j = 0; j < head_inner; ++j) {
        out_weight->data()[dst_offset + j] = weight_op->at(src_offset + j);
      }
    }
    std::vector<int64_t> out_shape(shape);
    out_shape[axis] /= head;
    auto new_type = RankedTensorType::get(out_shape, to_type);
    std::string suffix = base_name + "_head_" + std::to_string(idx);
    return top::WeightOp::create(op, suffix, *out_weight, new_type);
  } else {
    return top::NoneOp(op);
  }
}

Value create_matmul(PatternRewriter &rewriter, Value input, Value weight, Value bias,
                    int head, int idx, int axis, std::string name) {
  auto in_shape = module::getShape(input);
  auto type = module::getStorageType(input);
  auto op = input.getDefiningOp();
  auto none_op = module::getNoneOp(op);

  std::vector<NamedAttribute> attrs;
  auto weight_new = get_weight(weight, head, idx, axis, rewriter.getF32Type(), "weight");
  std::vector<Value> operands = {input, weight_new};
  if (!module::isNone(bias)) {
    auto bias_new = get_weight(bias, head, idx, 0, rewriter.getF32Type(), "bias");
    operands.push_back(bias_new);
  } else {
    operands.push_back(none_op);
  }
  auto weight_shape = module::getShape(weight_new);
  std::vector<int64_t> out_shape(in_shape);
  out_shape[2] = weight_shape[1];
  auto type_new = RankedTensorType::get(out_shape, type);
  std::string name_new = name + std::to_string(idx);
  auto name_loc = NameLoc::get(rewriter.getStringAttr(name_new));
  return rewriter.create<MatMulOp>(name_loc, type_new, operands, attrs);
}

struct TopFuseTransformer : public OpRewritePattern<TransformerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformerOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getHead() == 1) {
      return failure();
    }
    rewriter.setInsertionPointAfter(op);
    auto input = op.getInput();
    auto keys = op.getKeys();
    auto values = op.getValues();
    auto in_shape = module::getShape(input);
    auto k_shape = module::getShape(keys);
    // int64_t dim = in_shape.size();
    auto head = op.getHead();
    auto type = module::getStorageType(input);
    auto none_op = module::getNoneOp(op);
    auto musk_r = op.getMusk();
    std::string out_name = module::getName(op.getOutput()).data();
    std::vector<Value> operands;
    // attention for each head
    for (int i = 0; i < head; ++i) {

      // // queries
      // auto matmul_q = create_matmul(rewriter, input, op.getQueriesWeight(), op.getQueriesBias(),
      //                               head, i, 1, out_name + "_queries_");
      // std::vector<NamedAttribute> attrs_none;
      // // keys
      // auto matmul_k = create_matmul(rewriter, keys, op.getKeysWeight(), op.getKeysBias(),
      //                               head, i, 1, out_name + "_keys_");
      // // values
      // auto matmul_v = create_matmul(rewriter, values, op.getValuesWeight(), op.getValuesBias(),
      //                               head, i, 1, out_name + "_values_");
      // // q matmul k^T
      // std::vector<NamedAttribute> attrs_m0;
      // attrs_m0.push_back(rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
      // std::vector<int64_t> m_shape(in_shape);
      // m_shape[dim - 1] = k_shape[dim - 2];
      // auto type_m0 = RankedTensorType::get(m_shape, type);
      // std::string name_m0 = out_name + "_attention_matmul0_" + std::to_string(i);
      // auto name_loc_m0 = NameLoc::get(rewriter.getStringAttr(name_m0));
      // auto matmul_m0 =
      //     rewriter.create<MatMulOp>(name_loc_m0, type_m0,
      //                               ValueRange{matmul_q, matmul_k, none_op}, attrs_m0);
      // // mul scalar
      // std::vector<NamedAttribute> attrs_mul;
      // attrs_mul.push_back(rewriter.getNamedAttr("const_val", op.getScaleAttr()));
      // std::string name_mul = out_name + "_attention_scale_" + std::to_string(i);
      // auto name_loc_mul = NameLoc::get(rewriter.getStringAttr(name_mul));
      // auto mul = rewriter.create<MulConstOp>(name_loc_mul, type_m0,
      //                                        ValueRange{matmul_m0}, attrs_mul);
      // // expand dim
      // std::vector<NamedAttribute> attrs_sq;
      // attrs_sq.push_back(rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({2})));
      // std::vector<int64_t> r_shape(m_shape);
      // r_shape.insert(r_shape.begin() + 2, 1);
      // auto type_softmax = RankedTensorType::get(r_shape, type);
      // std::string name_exd = out_name + "_attention_expand_dim" + std::to_string(i);
      // auto name_loc_exd = NameLoc::get(rewriter.getStringAttr(name_exd));
      // auto expand_dim = rewriter.create<UnsqueezeOp>(name_loc_exd, type_softmax,
      //                                         ValueRange{mul}, attrs_sq);
      // // add musk
      // Value musk = expand_dim;
      // if (!module::isNone(op.getMusk())) {
      //   std::string name_musk = out_name + "_attention_musk_" + std::to_string(i);
      //   auto name_loc_musk = NameLoc::get(rewriter.getStringAttr(name_musk));
      //   musk = rewriter.create<AddOp>(name_loc_musk, type_softmax,
      //                                 ValueRange{expand_dim, op.getMusk()}, attrs_none);
      // }

      // // softmax
      // std::vector<NamedAttribute> attrs_soft;
      // attrs_soft.push_back(rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(dim)));
      // std::string name_s = out_name + "_attention_softmax_" + std::to_string(i);
      // auto name_loc_s = NameLoc::get(rewriter.getStringAttr(name_s));
      // auto softmax = rewriter.create<SoftmaxOp>(name_loc_s, type_softmax,
      //                                           ValueRange{musk}, attrs_soft);
      // // squeeze dim
      // std::string name_sqd = out_name + "_attention_squeeze_dim" + std::to_string(i);
      // auto name_loc_sqd = NameLoc::get(rewriter.getStringAttr(name_sqd));
      // auto squeeze_dim = rewriter.create<SqueezeOp>(name_loc_sqd, type_m0,
      //                                         ValueRange{softmax}, attrs_sq);
      // // matmul v
      // std::vector<int64_t> m1_shape(in_shape);
      // m1_shape[dim - 1] /= head;
      // auto type_m1 = RankedTensorType::get(m1_shape, type);
      // std::string name_m1 = out_name + "_attention_matmul1_" + std::to_string(i);
      // auto name_loc_m1 = NameLoc::get(rewriter.getStringAttr(name_m1));
      // auto matmul_m1 =
      //     rewriter.create<MatMulOp>(name_loc_m1, type_m1,
      //                               ValueRange{squeeze_dim, matmul_v, none_op}, attrs_none);
      // // multi head fuse
      // std::vector<Value> mat_operands;
      // auto weight_o = get_weight(op.getOutWeight(), head, i, 0, rewriter.getF32Type(), "_out_weight");
      // mat_operands.push_back(matmul_m1);
      // mat_operands.push_back(weight_o);
      // if (i == 0) {
      //   mat_operands.push_back(op.getOutBias());
      // } else {
      //   mat_operands.push_back(none_op);
      // }
      // std::string name_o = out_name + "_attention_out_" + std::to_string(i);
      // auto name_loc_o = NameLoc::get(rewriter.getStringAttr(name_o));
      // auto matmul_o =
      //     rewriter.create<MatMulOp>(name_loc_o, op.getOutput().getType(),
      //                               mat_operands, attrs_none);


      auto weight_q = get_weight(op.getQueriesWeight(), head, i, -1, rewriter.getF32Type(), "weight");
      auto weight_k = get_weight(op.getKeysWeight(), head, i, -1, rewriter.getF32Type(), "weight");
      auto weight_v = get_weight(op.getValuesWeight(), head, i, -1, rewriter.getF32Type(), "weight");
      auto weight_o = get_weight(op.getOutWeight(), head, i, -2, rewriter.getF32Type(), "weight");
      auto bias_q = get_weight(op.getQueriesBias(), head, i, -1, rewriter.getF32Type(), "bias");
      auto bias_k = get_weight(op.getKeysBias(), head, i, -1, rewriter.getF32Type(), "bias");
      auto bias_v = get_weight(op.getValuesBias(), head, i, -1, rewriter.getF32Type(), "bias");
      std::vector<Value> operands_t = {input, keys, values, weight_q, bias_q, weight_k, bias_k, weight_v, bias_v, weight_o};
      int64_t has_bias = module::isNone(op.getQueriesBias()) ? 0 : 1;
      has_bias |= module::isNone(op.getKeysBias()) ? 0 : 0x01<<1;
      has_bias |= module::isNone(op.getValuesBias()) ? 0 : 0x01<<2;
      if (i == 0 && !module::isNone(op.getOutBias())) {
        operands_t.push_back(op.getOutBias());
        has_bias |= 0x01<<3;
      } else {
        operands_t.push_back(none_op);
      }
      operands_t.push_back(op.getMusk());
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(1)));
      attrs.push_back(rewriter.getNamedAttr("scale", op.getScaleAttr()));
      attrs.push_back(rewriter.getNamedAttr("has_bias", rewriter.getI64IntegerAttr(has_bias)));
      std::string name_new = out_name + "_head_" + std::to_string(i);
      auto name_loc = NameLoc::get(rewriter.getStringAttr(name_new));
      auto attention = rewriter.create<TransformerOp>(name_loc, op.getOutput().getType(),
                                                operands_t, attrs);
      // multi head fuse
      operands.push_back(attention);
      if (i > 0) {
        std::vector<NamedAttribute> attrs_none;
        if (i != head - 1) {
          std::string name_add = out_name + "_attention_out_fuse_" + std::to_string(i);
          auto name_loc_add = NameLoc::get(rewriter.getStringAttr(name_add));
          auto mul = rewriter.create<AddOp>(name_loc_add, op.getOutput().getType(),
                                            operands, attrs_none);
          operands.clear();
          operands.push_back(mul);
        } else {
          auto mul = rewriter.create<AddOp>(op.getLoc(), op.getOutput().getType(),
                                            operands, attrs_none);
          op.replaceAllUsesWith(mul.getOperation());
          rewriter.eraseOp(op);
        }
      }
    }
    return success();
  }
};

void TransformerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<TopFuseTransformer>(context);
}
