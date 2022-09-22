//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

void top::MatMulOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                          bool asymmetric) {
  // refer quantize_convlike_layer_int8
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  int64_t batch, M, K, N;
  bool relu, with_bias;
  double relu_limit;
  parseParam(batch, M, K, N, with_bias, relu, relu_limit);
  assert(batch == 1); // only for fullyconnected now
  auto filterOp = cast<top::WeightOp>(right().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  int64_t in_zp, out_zp;
  double in_scale, out_scale;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp, asymmetric);

  double w_max = findMaxabs(filter_f32->data(), filter_f32->size());
  double w_scale = w_max / 127.0;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  for (uint64_t t = 0; t < filter_f32->size(); t++) {
    filter_int8->at(t) = Quant::to_int8(filter_f32->at(t) / w_scale);
  }

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(N);
  }

  for (int j = 0; j < N; j++) {
    int64_t bias_w_xz = 0;
    for (int i = 0; i < K; i++) {
      bias_w_xz += (int64_t)filter_int8->at(i * N + j) * in_zp;
    }

    if (with_bias) {
      bias_int32->data()[j] =
          std::round(bias_fp32->at(j) / (w_scale * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[j] = -bias_w_xz;
    }
  }
  with_bias = with_bias || in_zp != 0;
  int scale, shift;
  float scale_f = in_scale * w_scale / out_scale;
  get_scale_and_shift(scale_f, scale, shift, 32);

  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(shift)));
  attrs.push_back(
      rewriter.getNamedAttr("multiplier", rewriter.getSI32IntegerAttr(scale)));
  auto filter_type = right().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
  auto new_filter = WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(input());
  operands.push_back(new_filter);
  auto new_bias = bias();
  if (with_bias) {
    std::vector<int64_t> shape = {N};
    auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
    new_bias = WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    auto none = Module::getNoneOp(op);
    operands.push_back(none);
  }

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void top::MatMulOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::MatMulOp>(rewriter, getOperation());
}

void top::MatMulOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  operands.push_back(input());
  if (auto rightOp = dyn_cast<top::WeightOp>(right().getDefiningOp())) {
    operands.push_back(rightOp.clone_f16(op));
  }
  if (auto rightOp = dyn_cast<top::WeightOp>(bias().getDefiningOp())) {
    operands.push_back(rightOp.clone_f16(op));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto tensor_type = output().getType().cast<RankedTensorType>();
  auto newType =
      RankedTensorType::get(tensor_type.getShape(), rewriter.getF16Type());
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void top::MatMulOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  operands.push_back(input());
  if (auto rightOp = dyn_cast<top::WeightOp>(right().getDefiningOp())) {
    operands.push_back(rightOp.clone_bf16(op));
  }
  if (auto rightOp = dyn_cast<top::WeightOp>(bias().getDefiningOp())) {
    operands.push_back(rightOp.clone_bf16(op));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto tensor_type = output().getType().cast<RankedTensorType>();
  auto newType =
      RankedTensorType::get(tensor_type.getShape(), rewriter.getBF16Type());
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void top::MatMulOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  if (!Quant::isUniformQuantized(input(), right(), output())) {
    llvm_unreachable("input output should be quantized");
  }
  int64_t batch, M, K, N;
  bool relu, with_bias;
  double relu_limit;
  parseParam(batch, M, K, N, with_bias, relu, relu_limit);
  assert(batch == 1);
  auto input_qtype = Quant::getUniformQuantizedType(input());
  auto right_qtype = Quant::getUniformQuantizedType(right());
  auto output_qtype = Quant::getUniformQuantizedType(output());

  const double real_multiplier =
      input_qtype.getScale() * right_qtype.getScale() / output_qtype.getScale();
  int64_t multiplier, shift;
  QuantizeMultiplier(real_multiplier, &multiplier, &shift);
  int32_t right_zero_point = right_qtype.getZeroPoint();
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  std::vector<Value> operands;
  operands.push_back(input());
  auto right_stype = Module::getStorageType(right());
  auto right_new_type =
      RankedTensorType::get(Module::getShape(right()), right_stype);
  right().setType(right_new_type);
  operands.push_back(right());
  if (with_bias) {
    auto bias_stype = Module::getStorageType(bias());
    auto bias_new_type =
        RankedTensorType::get(Module::getShape(bias()), bias_stype);
    bias().setType(bias_new_type);
  }

  // std::string suffix = "_matmul";
  // std::string new_name = Module::getName(op).str() + suffix;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  if (right_zero_point)
    attrs.push_back(rewriter.getNamedAttr(
        "right_zp", rewriter.getI64IntegerAttr(right_zero_point)));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getI64IntegerAttr(-shift)));
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(ctx, tpu::RequantMode::TFlite_Lshift)));
  int32_t input_zeroPoint = input_qtype.getZeroPoint();

  if (input_zeroPoint != 0) {
    // merge input_zeroPoint to bias
    std::shared_ptr<std::vector<int32_t>> bias_quant;
    std::shared_ptr<std::vector<int8_t>> right_quant;
    right_quant = cast<top::WeightOp>(right().getDefiningOp()).read<int8_t>();
    if (isa<top::WeightOp>(bias().getDefiningOp())) {
      bias_quant = cast<top::WeightOp>(bias().getDefiningOp()).read<int32_t>();
    }
    auto right_type = right().getType().cast<RankedTensorType>();
    int64_t row_size = right_type.getShape()[0];
    int64_t col_size = right_type.getShape()[1];
    bias_quant->resize(col_size, 0);
    for (size_t r_ind = 0; r_ind < row_size; ++r_ind) {
      for (size_t c_ind = 0; c_ind < col_size; ++c_ind) {
        bias_quant->data()[c_ind] -=
            input_zeroPoint * right_quant->at(c_ind + r_ind * col_size);
      }
    }
    auto bias_type = RankedTensorType::get({col_size}, rewriter.getI32Type());
    auto new_bias = top::WeightOp::create(op, "MergedInputZeroPoint",
                                          *bias_quant, bias_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(bias());
  }
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, output().getType(), operands,
                                             attrs);
}
