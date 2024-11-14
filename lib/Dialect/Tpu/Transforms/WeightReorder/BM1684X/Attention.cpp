//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "ConvUtils.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace bm1684x;

template <typename T>
int64_t data_copy(top::WeightOp weight, int64_t offset,
                  std::shared_ptr<std::vector<T>> &new_weight) {
  auto data_fp32 = weight.read<T>();
  auto count = data_fp32->size();
  auto shape = module::getShape(weight);
  auto len = shape.size() == 2 ? align_up(shape[0], BM168x::NPU_NUM) * shape[1]
                               : count;
  memcpy(new_weight->data() + offset, data_fp32->data(), count * sizeof(T));
  return offset + len;
}

template <typename T>
void weight_reorder(T *dst, T *src, int dst_c, int dst_h, int dst_w,
                    int src_w) {
  for (int c = 0; c < dst_c; ++c) {
    for (int h = 0; h < dst_h; ++h) {
      T *dst_p = dst + (c * dst_h + h) * dst_w;
      T *src_p = src + (h * dst_c + c) * src_w;
      memcpy(dst_p, src_p, src_w * sizeof(T));
    }
  }
}

template <typename T>
Value weight_reorder(tpu::AttentionOp op, Type to_type, int N_q, int N_k,
                     int d) {
  auto q_w = op.getQueriesWeight().getDefiningOp<top::WeightOp>();
  auto k_w = op.getKeysWeight().getDefiningOp<top::WeightOp>();
  auto v_w = op.getValuesWeight().getDefiningOp<top::WeightOp>();
  int64_t weight_h =
      (align_up(N_q, BM168x::NPU_NUM) + align_up(N_k, BM168x::NPU_NUM) * 2);
  auto bytes = sizeof(T);
  auto EU_NUM = BM168x::eu_num(bytes);
  auto new_weight = std::make_shared<std::vector<T>>(weight_h * d);

  int offset = data_copy(q_w, 0, new_weight);
  offset = data_copy(k_w, offset, new_weight);
  offset = data_copy(v_w, offset, new_weight);
  auto new_weight1 =
      std::make_shared<std::vector<T>>(weight_h * align_up(d, EU_NUM));
  weight_reorder(new_weight1->data(), new_weight->data(), BM168x::NPU_NUM,
                 weight_h / BM168x::NPU_NUM, align_up(d, EU_NUM), d);

  std::vector<int64_t> weight_shape = {
      1, BM168x::NPU_NUM, weight_h / BM168x::NPU_NUM, align_up(d, EU_NUM)};
  auto new_type = RankedTensorType::get(weight_shape, to_type);
  auto new_op =
      top::WeightOp::create(op, "filter_reorder", *new_weight1, new_type);
  return new_op;
}

template <typename T>
Value bias_reorder(tpu::AttentionOp op, Type to_type, int N_q, int d) {
  auto q_bias = op.getQueriesBias();
  auto k_bias = op.getKeysBias();
  auto v_bias = op.getValuesBias();
  auto o_bias = op.getOutBias();
  int64_t bias_len = module::isNone(q_bias) ? 0 : d;
  bias_len += module::isNone(k_bias) ? 0 : d;
  bias_len += module::isNone(v_bias) ? 0 : d;
  bias_len += module::isNone(o_bias) ? 0 : N_q;
  if (bias_len) {
    int offset = 0;
    auto new_weight = std::make_shared<std::vector<T>>(bias_len);
    if (!module::isNone(q_bias)) {
      auto q_b = q_bias.getDefiningOp<top::WeightOp>();
      offset = data_copy(q_b, 0, new_weight);
    }
    if (!module::isNone(k_bias)) {
      auto k_b = k_bias.getDefiningOp<top::WeightOp>();
      offset = data_copy(k_b, offset, new_weight);
    }
    if (!module::isNone(v_bias)) {
      auto v_b = v_bias.getDefiningOp<top::WeightOp>();
      offset = data_copy(v_b, offset, new_weight);
    }
    if (!module::isNone(o_bias)) {
      auto o_b = o_bias.getDefiningOp<top::WeightOp>();
      offset = data_copy(o_b, offset, new_weight);
    }
    std::vector<int64_t> weight_shape = {1, 1, bias_len};
    auto new_type = RankedTensorType::get(weight_shape, to_type);
    auto new_op =
        top::WeightOp::create(op, "bias_reorder", *new_weight, new_type);
    return new_op;
  } else {
    return q_bias;
  }
}

template <typename T1, typename T2>
void attention_reorder(PatternRewriter &rewriter, tpu::AttentionOp op,
                       Type w_type, Type b_type) {
  auto none_op = module::getNoneOp(op);
  auto q_shape = module::getShape(op.getQueriesWeight());
  auto k_shape = module::getShape(op.getKeysWeight());

  auto new_op =
      weight_reorder<T1>(op, w_type, q_shape[0], k_shape[0], k_shape[1]);
  op->setOperand(3, new_op);
  auto bias_op = bias_reorder<T2>(op, b_type, q_shape[0], k_shape[1]);
  op->setOperand(4, bias_op);
  op->setOperand(5, none_op);
  op->setOperand(6, none_op);
  op->setOperand(7, none_op);
  op->setOperand(8, none_op);
  op->setOperand(10, none_op);
}

template <>
LogicalResult WeightReorder<tpu::AttentionOp, int8_t>::matchAndRewriteImpl(
    tpu::AttentionOp op, PatternRewriter &rewriter) const {

  if (!module::getStorageType(op.getInput()).isInteger(8))
    return failure();
  attention_reorder<int8_t, int32_t>(rewriter, op, rewriter.getI8Type(),
                                     rewriter.getI32Type());
  return success();
}

template <>
LogicalResult
WeightReorder<tpu::AttentionOp, BFloat16Type>::matchAndRewriteImpl(
    tpu::AttentionOp op, PatternRewriter &rewriter) const {

  if (!module::getStorageType(op.getInput()).isBF16())
    return failure();
  attention_reorder<uint16_t, uint16_t>(rewriter, op, rewriter.getBF16Type(),
                                        rewriter.getBF16Type());
  return success();
}

template <>
LogicalResult WeightReorder<tpu::AttentionOp, Float16Type>::matchAndRewriteImpl(
    tpu::AttentionOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getInput()).isF16())
    return failure();
  attention_reorder<uint16_t, uint16_t>(rewriter, op, rewriter.getF16Type(),
                                        rewriter.getF16Type());
  return success();
}
