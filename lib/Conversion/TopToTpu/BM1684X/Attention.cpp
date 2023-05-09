//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

const int64_t NPU_NUM = 64;

int64_t data_copy(top::WeightOp weight, int64_t offset,
                  std::shared_ptr<std::vector<float>> &new_weight) {
  auto data_fp32 = weight.read<float>();
  auto count = data_fp32->size();
  auto shape = module::getShape(weight);
  auto len = shape.size() == 2 ? align_up(shape[0], NPU_NUM) * shape[1] : count;
  memcpy(new_weight->data() + offset, data_fp32->data(), count*sizeof(float));
  return offset + len;
}

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

template <typename ElemTy>
Value lowering_attention_float(PatternRewriter &rewriter,
                              top::AttentionOp op) {
  auto stype = module::getStorageType(op.getInput());
  auto none_op = module::getNoneOp(op);
  if (op.getValues() == op.getKeys()) {
    op->setOperand(2, none_op);
  }
  if (op.getInput() == op.getKeys()) {
    op->setOperand(1, none_op);
  }
  auto q_shape = module::getShape(op.getQueriesWeight());
  auto k_shape = module::getShape(op.getKeysWeight());
  auto o_shape = module::getShape(op.getOutWeight());
  auto q_w = op.getQueriesWeight().getDefiningOp<top::WeightOp>();
  auto k_w = op.getKeysWeight().getDefiningOp<top::WeightOp>();
  auto v_w = op.getValuesWeight().getDefiningOp<top::WeightOp>();
  int64_t weight_h = (align_up(q_shape[0], NPU_NUM) + align_up(k_shape[0], NPU_NUM) + k_shape[0]);
  auto new_weight = std::make_shared<std::vector<float>>(weight_h * q_shape[1]);

  int offset = data_copy(q_w, 0, new_weight);
  offset = data_copy(k_w, offset, new_weight);
  offset = data_copy(v_w, offset, new_weight);

  std::vector<int64_t> weight_shape = {1, weight_h, q_shape[1]};
  auto new_type = RankedTensorType::get(weight_shape, stype);
  auto new_op =
          top::WeightOp::create(op, "filter_reorderd", *new_weight, new_type);
  op->setOperand(3, new_op);

  int64_t bias_len = module::isNone(op.getQueriesBias()) ? 0 : q_shape[1];
  bias_len += module::isNone(op.getKeysBias()) ? 0 : q_shape[1];
  bias_len += module::isNone(op.getValuesBias()) ? 0 : q_shape[1];
  bias_len += module::isNone(op.getOutBias()) ? 0 : o_shape[1];
  if (bias_len) {
    auto new_weight = std::make_shared<std::vector<float>>(bias_len);
    if (!module::isNone(op.getQueriesBias())) {
      auto q_b = op.getQueriesBias().getDefiningOp<top::WeightOp>();
      offset = data_copy(q_b, 0, new_weight);
    }
    if (!module::isNone(op.getKeysBias())) {
      auto k_b = op.getKeysBias().getDefiningOp<top::WeightOp>();
      offset = data_copy(k_b, offset, new_weight);
    }
    if (!module::isNone(op.getValuesBias())) {
      auto v_b = op.getValuesBias().getDefiningOp<top::WeightOp>();
      offset = data_copy(v_b, offset, new_weight);
    }
    if (!module::isNone(op.getOutBias())) {
      auto o_b = op.getOutBias().getDefiningOp<top::WeightOp>();
      offset = data_copy(o_b, offset, new_weight);
    }
    std::vector<int64_t> weight_shape = {1, 1, bias_len};
    auto new_type = RankedTensorType::get(weight_shape, stype);
    auto new_op =
          top::WeightOp::create(op, "bias_reorderd", *new_weight, new_type);
    op->setOperand(4, new_op);
  }

  op->setOperand(5, none_op);
  op->setOperand(6, none_op);
  op->setOperand(7, none_op);
  op->setOperand(8, none_op);
  op->setOperand(10, none_op);

  {
    auto shape = module::getShape(op.getOutWeight());
    std::vector<int64_t> weight_shape(shape);
    weight_shape.insert(weight_shape.begin(), 1);
    auto new_type = RankedTensorType::get(weight_shape, stype);
    op.getOutWeight().setType(new_type);
  }
  // lowering_common_float<tpu::AttentionOp, ElemTy>(rewriter, op);
  auto newType = getQuantFloatType<ElemTy>(op->getResult(0));
  auto nstype = module::getStorageType(newType);
  std::vector<Value> operands;
  int in_num_ops = op->getNumOperands();
  for (int i = 0; i < in_num_ops; ++i) {
    auto in = op->getOperand(i);
    if (isa<top::WeightOp>(in.getDefiningOp())) {
      auto wOp = in.getDefiningOp<top::WeightOp>();
      if (nstype.isF16()) {
        operands.push_back(wOp.clone_f16(op));
      } else if (nstype.isBF16()) {
        operands.push_back(wOp.clone_bf16(op));
      } else {
        operands.push_back(in);
      }
    } else {
      operands.push_back(in);
    }
  }
  auto attention = rewriter.replaceOpWithNewOp<tpu::AttentionOp>(op, newType,
                                                     operands, op->getAttrs());
  return attention.getOutput();
}

template <typename ElemTy>
void split_attention_head(PatternRewriter &rewriter, top::AttentionOp op) {
    auto input = op.getInput();
    auto keys = op.getKeys();
    auto values = op.getValues();
    rewriter.setInsertionPointAfterValue(input);
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
      auto weight_q = get_weight(op.getQueriesWeight(), head, i, -1, rewriter.getF32Type(), "weight");
      auto weight_k = get_weight(op.getKeysWeight(), head, i, -1, rewriter.getF32Type(), "weight");
      auto weight_v = get_weight(op.getValuesWeight(), head, i, -1, rewriter.getF32Type(), "weight");
      auto weight_o = get_weight(op.getOutWeight(), head, i, -2, rewriter.getF32Type(), "weight");
      auto bias_q = get_weight(op.getQueriesBias(), head, i, -1, rewriter.getF32Type(), "bias");
      auto bias_k = get_weight(op.getKeysBias(), head, i, -1, rewriter.getF32Type(), "bias");
      auto bias_v = get_weight(op.getValuesBias(), head, i, -1, rewriter.getF32Type(), "bias");
      std::vector<Value> operands_a = {input, keys, values, weight_q, bias_q, weight_k, bias_k, weight_v, bias_v, weight_o};
      int64_t has_bias = module::isNone(op.getQueriesBias()) ? 0 : 1;
      has_bias |= module::isNone(op.getKeysBias()) ? 0 : 0x01<<1;
      has_bias |= module::isNone(op.getValuesBias()) ? 0 : 0x01<<2;
      if (i == 0 && !module::isNone(op.getOutBias())) {
        operands_a.push_back(op.getOutBias());
        has_bias |= 0x01<<3;
      } else {
        operands_a.push_back(none_op);
      }
      operands_a.push_back(op.getMusk());
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(1)));
      attrs.push_back(rewriter.getNamedAttr("scale", op.getScaleAttr()));
      attrs.push_back(rewriter.getNamedAttr("has_bias", rewriter.getI64IntegerAttr(has_bias)));
      std::string name_new = out_name + "_head_" + std::to_string(i);
      auto name_loc = NameLoc::get(rewriter.getStringAttr(name_new));
      auto attention = rewriter.create<top::AttentionOp>(name_loc, op.getOutput().getType(),
                                                         operands_a, attrs);
      // multi head fuse
      operands.push_back(lowering_attention_float<ElemTy>(rewriter, attention));
      if (i > 0) {
        std::vector<NamedAttribute> attrs_none;
        auto newType = getQuantFloatType<ElemTy>(op->getResult(0));
        if (i != head - 1) {
          std::string name_add = out_name + "_attention_out_fuse_" + std::to_string(i);
          auto name_loc_add = NameLoc::get(rewriter.getStringAttr(name_add));
          auto mul = rewriter.create<tpu::AddOp>(name_loc_add, newType,
                                                 operands, attrs_none);
          operands.clear();
          operands.push_back(mul);
        } else {
          auto mul = rewriter.create<tpu::AddOp>(op.getLoc(), newType,
                                                 operands, attrs_none);
          rewriter.replaceOp(op, {mul.getOutput()});
        }
      }
    }
}

void AttentionLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::AttentionOp op) const {
  split_attention_head<Float16Type>(rewriter, op);
}
void AttentionLowering::LoweringINT4(PatternRewriter &rewriter, top::AttentionOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void AttentionLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::AttentionOp op, bool asymmetric) const {
  // llvm_unreachable("Not Implemented");
  split_attention_head<Float16Type>(rewriter, op);
}

void AttentionLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::AttentionOp op) const {
  split_attention_head<BFloat16Type>(rewriter, op);
}

void AttentionLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::AttentionOp op) const {
  split_attention_head<Float16Type>(rewriter, op);
}

void AttentionLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::AttentionOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
