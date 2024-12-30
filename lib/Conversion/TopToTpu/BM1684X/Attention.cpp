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

Value get_weight(Value weight, int head, int idx, int axis, Type to_type,
                 std::string base_name) {
  auto op = weight.getDefiningOp();
  if (module::isWeight(weight)) {
    auto shape = module::getShape(weight);
    auto dim = shape.size();
    axis = axis < 0 ? dim + axis : axis;
    int begin = shape[axis] / head * idx;
    int end = shape[axis] / head * (idx + 1);
    std::string suffix = base_name + "_head_" + std::to_string(idx);
    return dyn_cast<top::WeightOp>(op).split(begin, end, axis, to_type, suffix);
  } else {
    return top::NoneOp(op);
  }
}

top::AttentionOp attention_head(PatternRewriter &rewriter, top::AttentionOp op,
                                int index) {
  auto input = op.getInput();
  auto keys = op.getKeys();
  auto values = op.getValues();
  auto head = op.getHead();
  auto none_op = module::getNoneOp(op);
  std::string out_name = module::getName(op.getOutput()).data();
  // attention for each head
  auto weight_q =
      get_weight(op.getQueriesWeight(), head, index, -1,
                 module::getStorageType(op.getQueriesWeight()), "weight");
  auto weight_k =
      get_weight(op.getKeysWeight(), head, index, -1,
                 module::getStorageType(op.getKeysWeight()), "weight");
  auto weight_v =
      get_weight(op.getValuesWeight(), head, index, -1,
                 module::getStorageType(op.getValuesWeight()), "weight");
  auto weight_o =
      get_weight(op.getOutWeight(), head, index, -2,
                 module::getStorageType(op.getOutWeight()), "weight");
  auto bias_q = get_weight(op.getQueriesBias(), head, index, -1,
                           module::getStorageType(op.getQueriesBias()), "bias");
  auto bias_k = get_weight(op.getKeysBias(), head, index, -1,
                           module::getStorageType(op.getKeysBias()), "bias");
  auto bias_v = get_weight(op.getValuesBias(), head, index, -1,
                           module::getStorageType(op.getValuesBias()), "bias");
  std::vector<Value> operands_a = {input,  keys,     values, weight_q,
                                   bias_q, weight_k, bias_k, weight_v,
                                   bias_v, weight_o};
  int64_t has_bias = module::isNone(op.getQueriesBias()) ? 0 : 1;
  has_bias |= module::isNone(op.getKeysBias()) ? 0 : 0x01 << 1;
  has_bias |= module::isNone(op.getValuesBias()) ? 0 : 0x01 << 2;
  if (index == 0 && !module::isNone(op.getOutBias())) {
    operands_a.push_back(op.getOutBias());
    has_bias |= 0x01 << 3;
  } else {
    operands_a.push_back(none_op);
  }
  operands_a.push_back(op.getMask());
  int64_t dim = module::getShape(weight_q)[1];
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(1)));
  attrs.push_back(
      rewriter.getNamedAttr("dim", rewriter.getI64IntegerAttr(dim)));
  attrs.push_back(rewriter.getNamedAttr("scale", op.getScaleAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("has_bias", rewriter.getI64IntegerAttr(has_bias)));
  attrs.push_back(rewriter.getNamedAttr("scale_param", op.getScaleParamAttr()));
  std::string name_new = out_name + "_head_" + std::to_string(index);
  auto name_loc = NameLoc::get(rewriter.getStringAttr(name_new));
  auto attention = rewriter.create<top::AttentionOp>(
      name_loc, op.getOutput().getType(), operands_a, attrs);
  return attention;
}

void attention_reorder(PatternRewriter &rewriter, top::AttentionOp op) {
  auto none_op = module::getNoneOp(op);
  if (op.getValues() == op.getKeys()) {
    op->setOperand(2, none_op);
  }
  if (op.getInput() == op.getKeys()) {
    op->setOperand(1, none_op);
  }

  {
    auto shape = module::getShape(op.getOutWeight());
    std::vector<int64_t> weight_shape(shape);
    weight_shape.insert(weight_shape.begin(), 1);
    module::setShape(op.getOutWeight(), weight_shape);
  }
}

template <typename ElemTy>
Value lowering_attention_float(PatternRewriter &rewriter, top::AttentionOp op) {
  auto newType = getQuantFloatType<ElemTy>(op->getResult(0));
  auto nstype = module::getStorageType(newType);
  std::vector<Value> operands;
  int in_num_ops = op->getNumOperands();
  // bool bias_use_fp32 = module::isBM1688();
  for (int i = 0; i < in_num_ops; ++i) {
    auto in = op->getOperand(i);
    if (module::isWeight(in)) {
      auto wOp = in.getDefiningOp<top::WeightOp>();
      // if (i == 4 && bias_use_fp32) {
      //   operands.push_back(in);
      // } else if (nstype.isF16()) {
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
  operands.push_back(module::getNoneOp(op));
  auto attention = rewriter.replaceOpWithNewOp<tpu::AttentionOp>(
      op, newType, operands, op->getAttrs());
  return attention.getOutput();
}

template <typename ElemTy>
void lowering_multi_attention_float(PatternRewriter &rewriter,
                                    top::AttentionOp op) {
  rewriter.setInsertionPointAfter(op);
  auto head = op.getHead();
  std::string out_name = module::getName(op.getOutput()).data();
  std::vector<Value> operands;
  // attention for each head
  for (int i = 0; i < head; ++i) {
    auto attention = attention_head(rewriter, op, i);
    attention_reorder(rewriter, attention);
    // multi head fuse
    operands.push_back(lowering_attention_float<ElemTy>(rewriter, attention));
    if (i > 0) {
      std::vector<NamedAttribute> attrs_none;
      auto newType = getQuantFloatType<ElemTy>(op->getResult(0));
      if (i != head - 1) {
        std::string name_add =
            out_name + "_attention_out_fuse_" + std::to_string(i);
        auto name_loc_add = NameLoc::get(rewriter.getStringAttr(name_add));
        auto mul = rewriter.create<tpu::AddOp>(name_loc_add, newType, operands,
                                               attrs_none);
        operands.clear();
        operands.push_back(mul);
      } else {
        auto mul = rewriter.create<tpu::AddOp>(op.getLoc(), newType, operands,
                                               attrs_none);
        rewriter.replaceOp(op, {mul.getOutput()});
      }
    }
  }
}

template <typename T>
Value weight_quant(Value weight, float scale, std::string suffix,
                   Type to_type) {
  if (module::isNone(weight)) {
    return weight;
  }
  auto op = weight.getDefiningOp();
  std::shared_ptr<std::vector<float>> weight_fp32;
  auto weightOp = cast<top::WeightOp>(op);
  weight_fp32 = weightOp.read<float>();
  auto weight_int = std::make_shared<std::vector<T>>(weight_fp32->size());
  for (int64_t j = 0; j < weight_fp32->size(); j++) {
    weight_int->data()[j] = std::round(weight_fp32->at(j) / (scale));
  }
  auto filter_type = weight.getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(), to_type);
  return top::WeightOp::create(op, suffix, *weight_int, new_type);
}

Value generate_table(mlir::Operation *op, float scale) {
  std::vector<float> table(256, 0.0f);
  for (int i = 0; i < 256; ++i) {
    table[i] = std::exp(-1.0 * scale * i);
  }
  return create_lookup_table(op, table);
}

void generate_quant_param(std::vector<int64_t> &param, double scale) {
  int mul = 1, shift = 0;
  get_scale_and_shift(scale, mul, shift, 32);
  param.push_back(mul);
  param.push_back(shift);
  param.push_back(0);
}

double get_weight_sacle(Value weight) {
  auto wOp = dyn_cast<top::WeightOp>(weight.getDefiningOp());
  auto data_f32 = wOp.read<float>();
  double scale;
  if (wOp.getScale().has_value()) {
    auto weight_scale_v = module::getF64Array(wOp.getScale().value());
    scale = weight_scale_v->data()[0];
  } else {
    double w_max = findMaxabs(data_f32->data(), data_f32->size());
    scale = w_max / 127.0;
  }
  return scale;
}

Value lowering_attention_int(PatternRewriter &rewriter, top::AttentionOp op,
                             double ow_scale) {
  // get scale param
  auto scale_param = module::getF64Array(op.getScaleParam());
  double qo_scale = scale_param->at(0);
  double ko_scale = scale_param->at(1);
  double vo_scale = scale_param->at(2);
  double m0_scale = scale_param->at(3);
  double si_scale = scale_param->at(4);
  double so_scale = scale_param->at(5);
  double m1_scale = scale_param->at(6);
  int64_t zp = 0;
  double qw_scale = 1.f, q_scale = 1.f, kw_scale = 1.f, k_scale = 1.f;
  double vw_scale = 1.f, v_scale = 1.f, o_scale = 1.f;
  module::getScaleAndZeroPoint(op.getInput(), q_scale, zp, false);
  module::getScaleAndZeroPoint(op.getKeys(), k_scale, zp, false);
  module::getScaleAndZeroPoint(op.getValues(), v_scale, zp, false);
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, zp, false);
  qw_scale = get_weight_sacle(op.getQueriesWeight());
  kw_scale = get_weight_sacle(op.getKeysWeight());
  vw_scale = get_weight_sacle(op.getValuesWeight());
  // weight quantize
  Value q_w = weight_quant<int8_t>(op.getQueriesWeight(), qw_scale, "int8",
                                   rewriter.getI8Type());
  op->setOperand(3, q_w);
  Value q_b = weight_quant<int32_t>(op.getQueriesBias(), qw_scale * q_scale,
                                    "int32", rewriter.getI32Type());
  op->setOperand(4, q_b);
  Value k_w = weight_quant<int8_t>(op.getKeysWeight(), kw_scale, "int8",
                                   rewriter.getI8Type());
  op->setOperand(5, k_w);
  Value k_b = weight_quant<int32_t>(op.getKeysBias(), kw_scale * k_scale,
                                    "int32", rewriter.getI32Type());
  op->setOperand(6, k_b);
  Value v_w = weight_quant<int8_t>(op.getValuesWeight(), vw_scale, "int8",
                                   rewriter.getI8Type());
  op->setOperand(7, v_w);
  Value v_b = weight_quant<int32_t>(op.getValuesBias(), vw_scale * v_scale,
                                    "int32", rewriter.getI32Type());
  op->setOperand(8, v_b);
  Value o_w = weight_quant<int8_t>(op.getOutWeight(), ow_scale, "int8",
                                   rewriter.getI8Type());
  op->setOperand(9, o_w);
  Value o_b = weight_quant<int32_t>(op.getOutBias(), ow_scale * m1_scale,
                                    "int32", rewriter.getI32Type());
  op->setOperand(10, o_b);
  attention_reorder(rewriter, op);
  auto softmax_table = generate_table(op, si_scale);
  // generate requant param
  std::vector<int64_t> quant_param;
  // queries, keys, values, m0, m1, s_zp
  generate_quant_param(quant_param, qw_scale * q_scale / qo_scale);
  generate_quant_param(quant_param, kw_scale * k_scale / ko_scale);
  generate_quant_param(quant_param, vw_scale * v_scale / vo_scale);
  generate_quant_param(quant_param, qo_scale * ko_scale / m0_scale);
  generate_quant_param(quant_param, so_scale * vo_scale / m1_scale);
  quant_param.push_back(0);

  std::vector<Value> operands;
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto in = op->getOperand(i);
    operands.push_back(in);
  }
  operands.push_back(softmax_table);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(1)));
  attrs.push_back(rewriter.getNamedAttr("dim", op.getDimAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("scale", rewriter.getF64FloatAttr(so_scale)));
  attrs.push_back(rewriter.getNamedAttr("has_bias", op.getHasBiasAttr()));
  attrs.push_back(rewriter.getNamedAttr("quant_param",
                                        rewriter.getI64ArrayAttr(quant_param)));
  auto new_type = RankedTensorType::get(module::getShape(op.getOutput()),
                                        rewriter.getI32Type());
  auto attention = rewriter.replaceOpWithNewOp<tpu::AttentionOp>(
      op, new_type, operands, attrs);
  return attention.getOutput();
}

void lowering_multi_attention_int(PatternRewriter &rewriter,
                                  top::AttentionOp op) {
  rewriter.setInsertionPointAfter(op);
  auto head = op.getHead();
  std::string out_name = module::getName(op.getOutput()).data();
  std::vector<Value> operands;

  int multi = 1, shift = 0;
  int64_t zp;
  double o_scale;
  double ow_scale = get_weight_sacle(op.getOutWeight());
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, zp, false);
  auto scale_param = module::getF64Array(op.getScaleParam());
  double m1_scale = scale_param->at(6);
  get_scale_and_shift(m1_scale * ow_scale / o_scale, multi, shift, 32);
  // attention for each head
  for (int i = 0; i < head; ++i) {
    auto attention = attention_head(rewriter, op, i);
    // multi head fuse
    operands.push_back(lowering_attention_int(rewriter, attention, ow_scale));
    if (i > 0) {
      std::vector<NamedAttribute> attrs_none;
      auto newType = RankedTensorType::get(module::getShape(op.getOutput()),
                                           rewriter.getI32Type());
      std::string name_add =
          out_name + "_attention_out_fuse_" + std::to_string(i);
      auto name_loc_add = NameLoc::get(rewriter.getStringAttr(name_add));
      auto mul = rewriter.create<tpu::AddOp>(name_loc_add, newType, operands,
                                             attrs_none);
      operands.clear();
      operands.push_back(mul);
    }
  }
  // auto newType = RankedTensorType::get(module::getShape(op.getOutput()),
  //                                          rewriter.getI16Type());
  auto newType = getQuantInt8Type(op.getOutput());
  auto requant = do_requant(op.getLoc(), operands[0], newType, true, multi,
                            -shift, tpu::RequantMode::MultiplierShift);
  rewriter.replaceOp(op, {requant});
}

void AttentionLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::AttentionOp op) const {
  lowering_multi_attention_float<Float16Type>(rewriter, op);
}
void AttentionLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::AttentionOp op,
                                     bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void AttentionLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::AttentionOp op,
                                     bool asymmetric) const {
  lowering_multi_attention_int(rewriter, op);
}

void AttentionLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::AttentionOp op) const {
  lowering_multi_attention_float<BFloat16Type>(rewriter, op);
}

void AttentionLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::AttentionOp op) const {
  lowering_multi_attention_float<Float16Type>(rewriter, op);
}

void AttentionLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::AttentionOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void AttentionLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::AttentionOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
