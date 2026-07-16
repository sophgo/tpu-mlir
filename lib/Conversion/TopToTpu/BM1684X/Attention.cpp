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
  if (op->hasAttr("input_asym")) {
    attrs.push_back(
        rewriter.getNamedAttr("input_asym", rewriter.getBoolAttr(true)));
  }
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
    auto head_i_output = lowering_attention_float<ElemTy>(rewriter, attention);
    if (i == 0 && head == 1) {
      rewriter.replaceOp(op, head_i_output);
      return;
    }
    operands.push_back(head_i_output);
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

// Quantize a q/k/v (weight, bias) pair to int8/int32 while absorbing the input
// zero point into the bias (bias correction), mirroring MatMul.cpp.
// Returns the int8 weight, the int32 bias (None if no bias and zp == 0), and
// whether a bias is now present (a bias is synthesized when the original bias
// is absent but the input has a non-zero zero point).
struct QkvQuantResult {
  Value weight;
  Value bias;
  bool has_bias;
};
static QkvQuantResult quant_qkv_weight_bias(Operation *op, Value weight,
                                            Value bias, double w_scale,
                                            double in_scale, int64_t in_zp,
                                            Type i8_type, Type i32_type,
                                            const std::string &suffix) {
  // weight -> int8
  auto wOp = weight.getDefiningOp<top::WeightOp>();
  auto weight_fp32 = wOp.read<float>();
  auto weight_int8 = std::make_shared<std::vector<int8_t>>(weight_fp32->size());
  for (size_t i = 0; i < weight_fp32->size(); i++) {
    weight_int8->data()[i] = std::round(weight_fp32->at(i) / w_scale);
  }
  auto w_shape = module::getShape(weight);
  auto w_new_type = RankedTensorType::get(w_shape, i8_type);
  auto q_w =
      top::WeightOp::create(op, suffix + "_int8", *weight_int8, w_new_type);

  bool had_bias = !module::isNone(bias);
  if (!had_bias && in_zp == 0) {
    return {q_w, bias, false}; // keep None, unchanged behavior
  }
  // bias -> int32 with zero point correction: subtract
  //   sum_i(W_int8[i, j] * in_zp) from round(B / (w_scale * in_scale)).
  int64_t K = w_shape[0];
  int64_t N = w_shape[1];
  std::shared_ptr<std::vector<float>> bias_fp32;
  if (had_bias) {
    bias_fp32 = bias.getDefiningOp<top::WeightOp>().read<float>();
  }
  auto bias_int32 = std::make_shared<std::vector<int32_t>>(N);
  for (int64_t j = 0; j < N; j++) {
    int64_t bias_w_xz = 0;
    if (in_zp) {
      for (int64_t i = 0; i < K; i++) {
        bias_w_xz += (int64_t)weight_int8->at(i * N + j) * in_zp;
      }
    }
    double b_val = had_bias ? bias_fp32->at(j) : 0.0;
    bias_int32->data()[j] =
        std::round(b_val / (w_scale * in_scale) - (double)bias_w_xz);
  }
  std::vector<int64_t> b_shape =
      had_bias ? std::vector<int64_t>(module::getShape(bias).begin(),
                                      module::getShape(bias).end())
               : std::vector<int64_t>{N};
  auto b_new_type = RankedTensorType::get(b_shape, i32_type);
  auto q_b =
      top::WeightOp::create(op, suffix + "_int32", *bias_int32, b_new_type);
  return {q_w, q_b, true};
}

Value lowering_attention_int(PatternRewriter &rewriter, top::AttentionOp op,
                             double ow_scale, bool asymmetric) {
  // get scale param
  auto scale_param = module::getF64Array(op.getScaleParam());
  double qo_scale = scale_param->at(0);
  double ko_scale = scale_param->at(1);
  double vo_scale = scale_param->at(2);
  double m0_scale = scale_param->at(3);
  double si_scale = scale_param->at(4);
  double so_scale = scale_param->at(5);
  double m1_scale = scale_param->at(6);
  // Honor input zero point when either the lowering pass is asymmetric or the
  // op was marked "input_asym" by calibration (same OR-semantics as
  // MatMul/Conv). Internal tensors (qo/ko/vo, m0/si/so/m1) and the whole
  // output stay symmetric, so their zp is read with asymmetric == false.
  bool input_asymmetric = op->hasAttr("input_asym") || asymmetric;
  int64_t q_zp = 0, k_zp = 0, v_zp = 0, o_zp = 0;
  double qw_scale = 1.f, q_scale = 1.f, kw_scale = 1.f, k_scale = 1.f;
  double vw_scale = 1.f, v_scale = 1.f, o_scale = 1.f;
  module::getScaleAndZeroPoint(op.getInput(), q_scale, q_zp, input_asymmetric);
  module::getScaleAndZeroPoint(op.getKeys(), k_scale, k_zp, input_asymmetric);
  module::getScaleAndZeroPoint(op.getValues(), v_scale, v_zp, input_asymmetric);
  module::getScaleAndZeroPoint(op.getOutput(), o_scale, o_zp, false);
  qw_scale = get_weight_sacle(op.getQueriesWeight());
  kw_scale = get_weight_sacle(op.getKeysWeight());
  vw_scale = get_weight_sacle(op.getValuesWeight());
  // weight + bias quantize (with input zp bias correction for q/k/v)
  auto qkv = quant_qkv_weight_bias(
      op, op.getQueriesWeight(), op.getQueriesBias(), qw_scale, q_scale, q_zp,
      rewriter.getI8Type(), rewriter.getI32Type(), "weight_q");
  op->setOperand(3, qkv.weight);
  op->setOperand(4, qkv.bias);
  auto kkv = quant_qkv_weight_bias(
      op, op.getKeysWeight(), op.getKeysBias(), kw_scale, k_scale, k_zp,
      rewriter.getI8Type(), rewriter.getI32Type(), "weight_k");
  op->setOperand(5, kkv.weight);
  op->setOperand(6, kkv.bias);
  auto vkv = quant_qkv_weight_bias(
      op, op.getValuesWeight(), op.getValuesBias(), vw_scale, v_scale, v_zp,
      rewriter.getI8Type(), rewriter.getI32Type(), "weight_v");
  op->setOperand(7, vkv.weight);
  op->setOperand(8, vkv.bias);
  // output projection: input is the internal symmetric m1 tensor, no zp
  // correction needed.
  Value o_w = weight_quant<int8_t>(op.getOutWeight(), ow_scale, "int8",
                                   rewriter.getI8Type());
  op->setOperand(9, o_w);
  Value o_b = weight_quant<int32_t>(op.getOutBias(), ow_scale * m1_scale,
                                    "int32", rewriter.getI32Type());
  op->setOperand(10, o_b);
  // update has_bias if a bias was synthesized for q/k/v due to non-zero zp
  int64_t has_bias = op.getHasBias();
  if (qkv.has_bias) {
    has_bias |= 0x01 << 0;
  }
  if (kkv.has_bias) {
    has_bias |= 0x01 << 1;
  }
  if (vkv.has_bias) {
    has_bias |= 0x01 << 2;
  }
  op->setAttr("has_bias", rewriter.getI64IntegerAttr(has_bias));
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
                                  top::AttentionOp op, bool asymmetric) {
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
    operands.push_back(
        lowering_attention_int(rewriter, attention, ow_scale, asymmetric));
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
  lowering_multi_attention_int(rewriter, op, asymmetric);
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
