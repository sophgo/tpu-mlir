//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include <numeric>

using namespace mlir;
using namespace sophgo;
using namespace sophgo::top;
using namespace sophgo::helper;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "sophgo/Dialect/Top/IR/TopOpsDialect.cpp.inc"

void TopDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sophgo/Dialect/Top/IR/TopOps.cpp.inc"
      >();
  wFile = nullptr;
}

//===----------------------------------------------------------------------===//
// Top Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sophgo/Dialect/Top/IR/TopOps.cpp.inc"

void ConvOp::parseParam(int64_t &n, int64_t &ic, int64_t &ih, int64_t &iw,
                        int64_t &oc, int64_t &oh, int64_t &ow, int64_t &g,
                        int64_t &kh, int64_t &kw, int64_t &ins_h,
                        int64_t &ins_w, int64_t &sh, int64_t &sw, int64_t &pt,
                        int64_t &pb, int64_t &pl, int64_t &pr, int64_t &dh,
                        int64_t &dw, bool &is_dw, bool &with_bias,
                        bool &do_relu) {
  auto i_s = input().getType().cast<ShapedType>().getShape();
  auto k_s = filter().getType().cast<ShapedType>().getShape();
  auto o_s = output().getType().cast<ShapedType>().getShape();
  do_relu = this->do_relu();
  with_bias = !bias().getType().isa<NoneType>();
  n = i_s[0];
  ic = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oc = o_s[1];
  oh = o_s[2];
  ow = o_s[3];
  kh = kernel_shape().getValue()[0].cast<IntegerAttr>().getInt();
  kw = kernel_shape().getValue()[1].cast<IntegerAttr>().getInt();
  pt = pads().getValue()[0].cast<IntegerAttr>().getInt();
  pl = pads().getValue()[1].cast<IntegerAttr>().getInt();
  pb = pads().getValue()[2].cast<IntegerAttr>().getInt();
  pr = pads().getValue()[3].cast<IntegerAttr>().getInt();
  sh = strides().getValue()[0].cast<IntegerAttr>().getInt();
  sw = strides().getValue()[1].cast<IntegerAttr>().getInt();
  dh = dilations().getValue()[0].cast<IntegerAttr>().getInt();
  dw = dilations().getValue()[1].cast<IntegerAttr>().getInt();
  g = group();
  is_dw = (oc == ic && oc == g);
  return;
}

void MaxPoolOp::parseParam(int64_t &n, int64_t &c, int64_t &ih, int64_t &iw,
                           int64_t &oh, int64_t &ow, int64_t &kh, int64_t &kw,
                           int64_t &sh, int64_t &sw, int64_t &pt, int64_t &pb,
                           int64_t &pl, int64_t &pr, int64_t &pad_value,
                           bool &is_global, bool &count_include_pad) {
  auto i_s = input().getType().cast<ShapedType>().getShape();
  auto o_s = output().getType().cast<ShapedType>().getShape();

  kh = kernel_shape().getValue()[0].cast<IntegerAttr>().getInt();
  kw = kernel_shape().getValue()[1].cast<IntegerAttr>().getInt();
  sh = strides().getValue()[0].cast<IntegerAttr>().getInt();
  sw = strides().getValue()[1].cast<IntegerAttr>().getInt();

  size_t num_dims = i_s.size();
  assert(num_dims == 4); // 4 dims now
  n = i_s[0];
  c = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oh = o_s[2];
  ow = o_s[3];
  pt = pads().getValue()[0].cast<IntegerAttr>().getInt();
  pl = pads().getValue()[1].cast<IntegerAttr>().getInt();
  pb = pads().getValue()[2].cast<IntegerAttr>().getInt();
  pr = pads().getValue()[3].cast<IntegerAttr>().getInt();
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    is_global = true;
  }
  pad_value = this->pad_value();
  count_include_pad = this->count_include_pad();
}

void AvgPoolOp::parseParam(int64_t &n, int64_t &c, int64_t &ih, int64_t &iw,
                           int64_t &oh, int64_t &ow, int64_t &kh, int64_t &kw,
                           int64_t &sh, int64_t &sw, int64_t &pt, int64_t &pb,
                           int64_t &pl, int64_t &pr, int64_t &pad_value,
                           bool &is_global, bool &count_include_pad) {
  auto i_s = input().getType().cast<ShapedType>().getShape();
  auto o_s = output().getType().cast<ShapedType>().getShape();

  kh = kernel_shape().getValue()[0].cast<IntegerAttr>().getInt();
  kw = kernel_shape().getValue()[1].cast<IntegerAttr>().getInt();
  sh = strides().getValue()[0].cast<IntegerAttr>().getInt();
  sw = strides().getValue()[1].cast<IntegerAttr>().getInt();

  size_t num_dims = i_s.size();
  assert(num_dims == 4); // 4 dims now
  n = i_s[0];
  c = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oh = o_s[2];
  ow = o_s[3];
  pt = pads().getValue()[0].cast<IntegerAttr>().getInt();
  pl = pads().getValue()[1].cast<IntegerAttr>().getInt();
  pb = pads().getValue()[2].cast<IntegerAttr>().getInt();
  pr = pads().getValue()[3].cast<IntegerAttr>().getInt();
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    is_global = true;
  }
  pad_value = this->pad_value();
  count_include_pad = this->count_include_pad();
}

void MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K, int64_t &N,
                          bool &with_bias) {
  auto i_s = input().getType().cast<ShapedType>().getShape();
  auto r_s = right().getType().cast<ShapedType>().getShape();
  auto o_s = output().getType().cast<ShapedType>().getShape();
  with_bias = !bias().getType().isa<mlir::NoneType>();
  auto r_dims = r_s.size();
  auto i_dims = i_s.size();
  N = r_s[r_dims - 1];
  K = r_s[r_dims - 2];
  if (r_dims > 2) {
    M = i_s[i_dims - 2];
    assert(i_s[i_dims - 1] == K);
    batch = std::accumulate(r_s.begin(), r_s.begin() + r_dims - 2, 1,
                            std::multiplies<int64_t>());
  } else {
    batch = 1;
    M = std::accumulate(i_s.begin(), i_s.begin() + i_dims - 1, 1,
                        std::multiplies<int64_t>());
  }
}

template <typename T> std::shared_ptr<std::vector<T>> WeightOp::read() {
  auto op = getOperation();
  auto dialect = op->getDialect();
  auto topDialect = llvm::cast<TopDialect>(dialect);
  if (topDialect->wFile == nullptr) {
    auto moduleOp = Module::getModuleOp(op);
    auto weight_file = Module::getWeightFile(moduleOp);
    topDialect->loadWeightFile(weight_file);
  }
  auto type = output().getType().cast<ShapedType>();
  return topDialect->wFile->readTensor<T>(name(), type);
}

std::shared_ptr<std::vector<float>> WeightOp::read_as_float() {
  auto type = getType().cast<ShapedType>();
  auto dtype = type.getElementType();
  if (dtype.isInteger(8)) {
    auto data_i8 = read<int8_t>();
    return std::make_shared<std::vector<float>>(data_i8->begin(),
                                                data_i8->end());
  } else if (dtype.isF32()) {
    return read<float>();
  } else if (Quant::isUniformQuantized(output())) {
    auto data_i8 = read<int8_t>();
    return std::make_shared<std::vector<float>>(data_i8->begin(),
                                                data_i8->end());
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    return std::make_shared<std::vector<float>>(data_i16->begin(),
                                                data_i16->end());
  } else if (dtype.isInteger(32)) {
    auto data_i32 = read<int32_t>();
    return std::make_shared<std::vector<float>>(data_i32->begin(),
                                                data_i32->end());
  }
  dump();
  llvm_unreachable("weight data not support read now");
  return nullptr;
}
std::shared_ptr<std::vector<uint8_t>> WeightOp::read_as_byte() {
  auto type = getType().cast<ShapedType>();
  auto dtype = type.getElementType();
  if (dtype.isInteger(8)) {
    return read<uint8_t>();
  } else if (dtype.isF32()) {
    auto data_f32 = read<float>();
    auto bytes = data_f32->size() * sizeof(float);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_f32->data(), bytes);
    return data_u8;
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    auto bytes = data_i16->size() * sizeof(int16_t);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_i16->data(), bytes);
    return data_u8;
  }
  dump();
  llvm_unreachable("weight data not support read now");
  return nullptr;
}

template <typename T>
Value WeightOp::create(Operation *OwnerOp, llvm::StringRef suffix,
                       const std::vector<T> &data, RankedTensorType &type) {
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  auto dialect = ctx->getLoadedDialect("top");
  auto topDialect = llvm::cast<TopDialect>(dialect);
  if (topDialect->wFile == nullptr) {
    auto moduleOp = Module::getModuleOp(OwnerOp);
    auto weight_file = Module::getWeightFile(moduleOp);
    topDialect->loadWeightFile(weight_file);
  }
  std::string op_name = Module::getName(OwnerOp).str();
  std::string new_name = op_name + "_" + suffix.str();
  auto ret = topDialect->wFile->addTensor(new_name, &data, type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(OwnerOp->getLoc(), type, nameAttr);
  return newOp.getResult();
}

template std::shared_ptr<std::vector<float>> WeightOp::read();
template std::shared_ptr<std::vector<int8_t>> WeightOp::read();
template std::shared_ptr<std::vector<int16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint8_t>> WeightOp::read();
template std::shared_ptr<std::vector<int32_t>> WeightOp::read();
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<float> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int16_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int8_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint8_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int32_t> &data,
                                RankedTensorType &type);
