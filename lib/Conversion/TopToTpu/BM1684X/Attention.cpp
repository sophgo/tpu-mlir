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

template <typename ElemTy>
void lowering_attention_float(PatternRewriter &rewriter,
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
  int64_t weight_h = (align_up(q_shape[0], NPU_NUM) + align_up(k_shape[0], NPU_NUM) * 2);
  auto new_weight = std::make_shared<std::vector<float>>(weight_h * q_shape[1]);

  int offset = data_copy(q_w, 0, new_weight);
  offset = data_copy(k_w, offset, new_weight);
  offset = data_copy(v_w, offset, new_weight);

  std::vector<int64_t> weight_shape = {1, weight_h, q_shape[1]};
  auto new_type = RankedTensorType::get(weight_shape, stype);
  auto new_op =
          top::WeightOp::create(op, "filter_reorderd", *new_weight, new_type);
  op->setOperand(3, new_op);

  int64_t bias_len = module::isNone(op.getQueriesBias()) ? 0 : 1;
  bias_len += module::isNone(op.getKeysBias()) ? 0 : 1;
  bias_len += module::isNone(op.getValuesBias()) ? 0 : 1;
  if (bias_len) {
    auto new_weight = std::make_shared<std::vector<float>>(bias_len * q_shape[1]);
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
    std::vector<int64_t> weight_shape = {1, 1, bias_len * q_shape[1]};
    auto new_type = RankedTensorType::get(weight_shape, stype);
    auto new_op =
          top::WeightOp::create(op, "bias_reorderd", *new_weight, new_type);
    op->setOperand(4, new_op);
  }

  op->setOperand(5, none_op);
  op->setOperand(6, none_op);
  op->setOperand(7, none_op);
  op->setOperand(8, none_op);

  if (!module::isNone(op.getOutBias())) {
    auto new_weight_o = std::make_shared<std::vector<float>>((align_up(o_shape[0], NPU_NUM) + 1) * o_shape[1]);
    auto o_w = op.getOutWeight().getDefiningOp<top::WeightOp>();
    offset = data_copy(o_w, 0, new_weight_o);
    auto o_b = op.getOutBias().getDefiningOp<top::WeightOp>();
    offset = data_copy(o_b, offset, new_weight_o);
    std::vector<int64_t> weight_shape = {1, (align_up(o_shape[0], NPU_NUM) + 1), o_shape[1]};
    auto new_type = RankedTensorType::get(weight_shape, stype);
    auto new_op =
            top::WeightOp::create(op, "filter_o_reorderd", *new_weight_o, new_type);
    op->setOperand(9, new_op);
    op->setOperand(10, none_op);
  } else {
    auto shape = module::getShape(op.getOutWeight());
    std::vector<int64_t> weight_shape(shape);
    weight_shape.insert(weight_shape.begin(), 1);
    auto new_type = RankedTensorType::get(weight_shape, stype);
    op.getOutWeight().setType(new_type);
  }
  lowering_common_float<tpu::AttentionOp, ElemTy>(rewriter, op);
}

void AttentionLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::AttentionOp op) const {
  lowering_attention_float<Float16Type>(rewriter, op);
}
void AttentionLowering::LoweringINT4(PatternRewriter &rewriter, top::AttentionOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}
void AttentionLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::AttentionOp op, bool asymmetric) const {
  llvm_unreachable("Not Implemented");
}

void AttentionLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::AttentionOp op) const {
  lowering_attention_float<BFloat16Type>(rewriter, op);
}

void AttentionLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::AttentionOp op) const {
  lowering_attention_float<Float16Type>(rewriter, op);
}

void AttentionLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::AttentionOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
