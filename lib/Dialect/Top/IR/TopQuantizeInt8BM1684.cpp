//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Interfaces/QuantizeInterface.h"
#include "sophgo/Support/Dnnl/Conv.h"
#include "sophgo/Support/Dnnl/Pool.h"
#include "sophgo/Support/Dnnl/MatMul.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;

static const double QMAX_INT8 = 127.0;
static const int BITS_INT8 = 8;

Value top::ConvOp::quantize_int8_bm1684() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  operands.push_back(input());
  std::vector<NamedAttribute> attrs;
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  auto th_input = Quant::getThreshold(input());
  auto th_output = Quant::getThreshold(output());
  auto filter_max = findMaxabs(filter_f32->data(), filter_f32->size());
  int rshift = calRightShiftNum(filter_max, th_input, th_output, BITS_INT8);
  rshift = std::max(rshift, 0);
  std::shared_ptr<std::vector<int16_t>> bias_int16;
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    auto bias_fp32 = biasOp.read<float>();
    float bias_scale = 1.0 * (1 << rshift) * QMAX_INT8 / th_output;
    int bias_len = bias_fp32->size();
    bias_int16 = std::make_shared<std::vector<int16_t>>(bias_len);
    float overflow_ratio = quantizeToInt16(
        bias_fp32->data(), bias_int16->data(), bias_len, bias_scale);

    int rightShiftDec = 2;
    while (overflow_ratio > 0.03 && rshift > 0) {
      rshift--;
      bias_scale = 1.0 * (1 << rshift) * QMAX_INT8 / th_output;
      overflow_ratio = quantizeToInt16(bias_fp32->data(), bias_int16->data(),
                                       bias_len, bias_scale);
      rightShiftDec--;
    }
  }
  float scale = 1.0 * (1 << rshift) * th_input / th_output;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  quantizeToInt8(filter_f32->data(), filter_int8->data(), filter_f32->size(),
                 scale);
  auto filter_type = filter().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), builder.getI8Type());
  auto new_filter = WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(new_filter);
  Value new_bias = bias();
  if (with_bias) {
    auto bias_type = bias().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(bias_type.getShape(), builder.getIntegerType(16));
    new_bias = WeightOp::create(op, "bias_int16", *bias_int16, new_type);
  }
  operands.push_back(new_bias);
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
  auto newOp = builder.create<tpu::ConvOp>(op->getLoc(), output().getType(),
                                           ArrayRef<Value>{operands},
                                           ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output());
  return newOp.output();
}

Value top::ReluOp::quantize_int8_bm1684() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::ReluOp>(op->getLoc(), output().getType(),
                                           ArrayRef<Value>{operands},
                                           ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output());
  return newOp.output();
}

Value top::AddOp::quantize_int8_bm1684() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs);
  std::vector<double> coeff_v(nInputs, 1.0);
  auto th_output = Quant::getThreshold(output());

  if (coeff().hasValue()) {
    for (auto v : coeff().getValue()) {
      coeff_v.push_back(v.cast<FloatAttr>().getValueAsDouble());
    }
  }

  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    auto th_input = Quant::getThreshold(input);
    rshift_v[i] =
        calRightShiftNumUseCblas(coeff_v[i], th_input, th_output, BITS_INT8);
    float scale = 1.0 * (1 << rshift_v[i]) * th_input / th_output;
    int8_t multiplier_int8 = 0;
    float coeff = coeff_v[i];
    quantizeToInt8(&coeff, &multiplier_int8, 1, scale);
    coeff_v[i] = (double)multiplier_int8;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(
      builder.getNamedAttr("coeff", builder.getF64ArrayAttr(coeff_v)));
  attrs.push_back(
      builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), output().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output());
  return newOp.output();
}

Value top::MaxPoolOp::quantize_int8_bm1684() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::MaxPoolOp>(op->getLoc(), output().getType(),
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output());
  return newOp.output();
}

Value top::AvgPoolOp::quantize_int8_bm1684() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::AvgPoolOp>(op->getLoc(), output().getType(),
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output());
  return newOp.output();
}

Value top::ReshapeOp::quantize_int8_bm1684() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::ReshapeOp>(op->getLoc(), output().getType(),
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output());
  return newOp.output();
}

Value top::MatMulOp::quantize_int8_bm1684() {
  // refer quantize_convlike_layer_int8
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  int64_t batch, M, K, N;
  bool with_bias;
  parseParam(batch, M, K, N, with_bias);
  assert(batch == 1); // only for fullyconnected now
  const int nInputs = op->getNumOperands();
  auto th_output = Quant::getThreshold(output());
  auto th_input = Quant::getThreshold(input());
  auto filterOp = cast<top::WeightOp>(right().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  double filter_max = findMaxabs(filter_f32->data(), filter_f32->size());
  int rshift = calRightShiftNum(filter_max, th_input, th_output, BITS_INT8);
  rshift = rshift >= 0 ? rshift : 0;
  std::shared_ptr<std::vector<int16_t>> bias_int16;
  if (with_bias) {
    float bias_scale = 1.0 * (1 << rshift) * QMAX_INT8 / th_output;
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    auto bias_f32 = biasOp.read<float>();
    bias_int16 = std::make_shared<std::vector<int16_t>>(bias_f32->size());
    float overflow_ratio = quantizeToInt16(bias_f32->data(), bias_int16->data(),
                                           bias_f32->size(), bias_scale);

    while (overflow_ratio > 0.03 && rshift > 0) {
      rshift--;
      bias_scale = 1.0 * (1 << rshift) * QMAX_INT8 / th_output;
      overflow_ratio = quantizeToInt16(bias_f32->data(), bias_int16->data(),
                                       bias_f32->size(), bias_scale);
    }
  }
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
  float scale = 1.0 * (1 << rshift) * th_input / th_output;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  quantizeToInt8(filter_f32->data(), filter_int8->data(), filter_f32->size(),
                 scale);
  auto filter_type = right().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), builder.getI8Type());
  auto new_filter = WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(input());
  operands.push_back(new_filter);
  auto new_bias = bias();
  if (with_bias) {
    auto bias_type = bias().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(bias_type.getShape(), builder.getIntegerType(16));
    new_bias = WeightOp::create(op, "bias_int16", *bias_int16, new_type);
  }
  operands.push_back(new_bias);
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::MatMulOp>(op->getLoc(), output().getType(),
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output());
  return newOp.output();
}
