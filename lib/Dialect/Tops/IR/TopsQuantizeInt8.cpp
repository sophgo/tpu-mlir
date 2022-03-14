//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Interfaces/QuantizeInterface.h"
#include "sophgo/Support/DnnlConv.h"
#include "sophgo/Support/DnnlPool.h"
#include "sophgo/Support/DnnlMatMul.h"
#include "sophgo/Support/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "dnnl.hpp"

using namespace sophgo;
using namespace mlir;

Value tops::ConvOp::quantize_int8() {
  // int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
  //     pl, pr, dh, dw;
  // bool is_dw, with_bias, relu;
  // parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt,
  // pb,
  //            pl, pr, dh, dw, is_dw, with_bias, relu);
  // auto filterOp = cast<tops::WeightOp>(filter().getDefiningOp());
  // auto biasOp = cast<tops::WeightOp>(bias().getDefiningOp());
  // auto filter = filterOp.read<float>();
  // auto bias = biasOp.read<float>();
  // auto type = output().getType().cast<RankedTensorType>();
  // auto fmax = findMaxabs(filter->data(), filter->size());

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
  auto newOp = builder.create<tpu::ConvOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value tops::ReluOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::ReluOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value tops::AddOp::quantize_int8() {
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
  auto newOp = builder.create<tpu::AddOp>(
      op->getLoc(), getResult().getType(), ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
  return newOp.getResult();
}

Value tops::MaxPoolOp::quantize_int8() {
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

Value tops::AvgPoolOp::quantize_int8() {
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

Value tops::ReshapeOp::quantize_int8() {
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

Value tops::MatMulOp::quantize_int8() {
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
