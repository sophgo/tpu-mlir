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

void MatMulLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  lowering_common_float<tpu::MatMulOp>(rewriter, op);
}

void MatMulLowering::LoweringINT8(PatternRewriter &rewriter, top::MatMulOp op,
                                  bool asymmetric) const {
  // refer quantize_convlike_layer_int8
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  int64_t batch, M, K, N;
  bool relu, with_bias;
  double relu_limit;
  op.parseParam(batch, M, K, N, with_bias, relu, relu_limit);
  assert(batch == 1); // only for fullyconnected now
  auto filterOp = cast<top::WeightOp>(op.right().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  int64_t in_zp, out_zp;
  double in_scale, out_scale;
  Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(op.output(), out_scale, out_zp, asymmetric);

  double w_max = findMaxabs(filter_f32->data(), filter_f32->size());
  double w_scale = w_max / 127.0;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  for (uint64_t t = 0; t < filter_f32->size(); t++) {
    filter_int8->at(t) = Quant::to_int8(filter_f32->at(t) / w_scale);
  }

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(op.bias().getDefiningOp());
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
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(shift)));
  attrs.push_back(
      rewriter.getNamedAttr("multipliers", rewriter.getI64ArrayAttr(scale)));
  auto filter_type = op.right().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
  auto new_filter =
      top::WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(op.input());
  operands.push_back(new_filter);
  auto new_bias = op.bias();
  if (with_bias) {
    std::vector<int64_t> shape = {N};
    auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
    new_bias = top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    auto none = Module::getNoneOp(op);
    operands.push_back(none);
  }

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void MatMulLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::MatMulOp op) const {
  auto ctx = getContext();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  operands.push_back(op.input());
  if (auto rightOp = dyn_cast<top::WeightOp>(op.right().getDefiningOp())) {
    operands.push_back(rightOp.clone_bf16(op));
  }
  if (auto biasOp = dyn_cast<top::WeightOp>(op.bias().getDefiningOp())) {
    operands.push_back(biasOp.clone_bf16(op));
  } else {
    auto none = Module::getNoneOp(op);
    operands.push_back(none);
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto tensor_type = op.output().getType().cast<RankedTensorType>();
  auto newType =
      RankedTensorType::get(tensor_type.getShape(), rewriter.getBF16Type());
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void MatMulLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  auto ctx = getContext();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  operands.push_back(op.input());
  if (auto rightOp = dyn_cast<top::WeightOp>(op.right().getDefiningOp())) {
    operands.push_back(rightOp.clone_f16(op));
  }
  if (auto biasOp = dyn_cast<top::WeightOp>(op.bias().getDefiningOp())) {
    operands.push_back(biasOp.clone_f16(op));
  } else {
    auto none = Module::getNoneOp(op);
    operands.push_back(none);
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto tensor_type = op.output().getType().cast<RankedTensorType>();
  auto newType =
      RankedTensorType::get(tensor_type.getShape(), rewriter.getF16Type());
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void MatMulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::MatMulOp op) const {
  if (!Quant::isUniformQuantized(op.input(), op.right(), op.output())) {
    llvm_unreachable("input output should be quantized");
  }
  int64_t batch, M, K, N;
  bool relu, with_bias;
  double relu_limit;
  op.parseParam(batch, M, K, N, with_bias, relu, relu_limit);
  assert(batch == 1);
  auto input_qtype = Quant::getUniformQuantizedType(op.input());
  auto right_qtype = Quant::getUniformQuantizedType(op.right());
  auto output_qtype = Quant::getUniformQuantizedType(op.output());

  const double real_multiplier =
      input_qtype.getScale() * right_qtype.getScale() / output_qtype.getScale();
  int64_t multiplier, shift;
  QuantizeMultiplier(real_multiplier, &multiplier, &shift);
  int32_t right_zero_point = right_qtype.getZeroPoint();
  auto ctx = getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  operands.push_back(op.input());
  auto right_stype = Module::getStorageType(op.right());
  auto right_new_type =
      RankedTensorType::get(Module::getShape(op.right()), right_stype);
  op.right().setType(right_new_type);
  operands.push_back(op.right());
  if (with_bias) {
    auto bias_stype = Module::getStorageType(op.bias());
    auto bias_new_type =
        RankedTensorType::get(Module::getShape(op.bias()), bias_stype);
    op.bias().setType(bias_new_type);
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
      "multipliers", rewriter.getI64ArrayAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(-shift)));
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(ctx, tpu::RequantMode::TFlite_Lshift)));
  int32_t input_zeroPoint = input_qtype.getZeroPoint();

  if (input_zeroPoint != 0) {
    // merge input_zeroPoint to bias
    std::shared_ptr<std::vector<int32_t>> bias_quant;
    std::shared_ptr<std::vector<int8_t>> right_quant;
    right_quant =
        cast<top::WeightOp>(op.right().getDefiningOp()).read<int8_t>();
    if (isa<top::WeightOp>(op.bias().getDefiningOp())) {
      bias_quant =
          cast<top::WeightOp>(op.bias().getDefiningOp()).read<int32_t>();
    }
    auto right_type = op.right().getType().cast<RankedTensorType>();
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
    operands.push_back(op.bias());
  }
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, op.output().getType(),
                                             operands, attrs);
}

} // namespace bm1684x
} // namespace tpu_mlir
