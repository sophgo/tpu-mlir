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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"

using namespace sophgo;
using namespace mlir;

template <typename T> static T getQuantType(Value v) {
  return v.getType().cast<RankedTensorType>().getElementType().cast<T>();
}

static double getThreshold(Value v) {
  auto type = getQuantType<quant::CalibratedQuantizedType>(v);
  assert(type.getMax() == -type.getMin());
  return type.getMax();
}

static double getMax(Value v) {
  auto type = getQuantType<quant::CalibratedQuantizedType>(v);
  return type.getMax();
}

static double getMin(Value v) {
  auto type = getQuantType<quant::CalibratedQuantizedType>(v);
  return type.getMin();
}

const double qmax8bit = 127.0;

Value top::ConvOp::quantize_int8() {
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
  auto filter_fp32 = filterOp.read<float>();
  auto th_input = getThreshold(input());
  auto th_output = getThreshold(output());
  auto filter_max = findMaxabs(filter_fp32->data(), filter_fp32->size());
  int rshift = calRightShiftNum(filter_max, th_input, th_output, 8);
  rshift = std::max(rshift, 0);
  Value new_bias = bias();
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    auto bias_fp32 = biasOp.read<float>();
    float bias_scale = 1.0 * (1 << rshift) * qmax8bit / th_output;
    int bias_len = bias_fp32->size();
    auto bias_int16 = std::make_shared<std::vector<int16_t>>(bias_len);
    float overflow_ratio = quantizeToInt16(
        bias_fp32->data(), bias_int16->data(), bias_len, bias_scale);

    int rightShiftDec = 2;
    while (overflow_ratio > 0.03 && rshift > 0) {
      rshift--;
      bias_scale = 1.0 * (1 << rshift) * qmax8bit / th_output;
      overflow_ratio = quantizeToInt16(bias_fp32->data(), bias_int16->data(),
                                       bias_len, bias_scale);
      rightShiftDec--;
    }
    overflow_ratio = quantizeToInt16(bias_fp32->data(), bias_int16->data(),
                                     bias_len, bias_scale);
    auto bias_type = bias().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(bias_type.getShape(), builder.getIntegerType(16));
    new_bias = WeightOp::create(op, "bias_int16", *bias_int16,
                                new_type);
  }
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
  float scale = 1.0 * (1 << rshift) * th_input / th_output;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_fp32->size());
  quantizeToInt8(filter_fp32->data(), filter_int8->data(), filter_fp32->size(),
                 scale);
  auto filter_type = filter().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), builder.getI8Type());
  auto new_filter =
      WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(new_filter);
  operands.push_back(new_bias);
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::ConvOp>(op->getLoc(), getResult().getType(),
                                           ArrayRef<Value>{operands},
                                           ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value top::ReluOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::ReluOp>(op->getLoc(), getResult().getType(),
                                           ArrayRef<Value>{operands},
                                           ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value top::AddOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), getResult().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value top::MaxPoolOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::MaxPoolOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value top::AvgPoolOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::AvgPoolOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value top::ReshapeOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::ReshapeOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value top::MatMulOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::MatMulOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}
