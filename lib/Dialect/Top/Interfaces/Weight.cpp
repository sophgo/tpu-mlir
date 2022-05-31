//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/fp16_bf16.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::top;
using namespace mlir;

template <typename T> std::shared_ptr<std::vector<T>> WeightOp::read() {
  auto op = getOperation();
  auto dialect = op->getDialect();
  auto topDialect = llvm::cast<TopDialect>(dialect);
  if (topDialect->wFile == nullptr) {
    auto moduleOp = Module::getModuleOp(op);
    auto weight_file = Module::getWeightFile(moduleOp);
    topDialect->loadWeightFile(weight_file);
  }
  auto type = output().getType().cast<RankedTensorType>();
  return topDialect->wFile->readTensor<T>(name(), type);
}

std::shared_ptr<std::vector<float>> WeightOp::read_as_float() {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  if (dtype.isInteger(8)) {
    auto data_i8 = read<int8_t>();
    return std::make_shared<std::vector<float>>(data_i8->begin(),
                                                data_i8->end());
  } else if (dtype.isF32()) {
    return read<float>();
  } else if (dtype.isF16()) {
    auto data_u16 = read<uint16_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (int i = 0; i < data_u16->size(); i++) {
      data_f32->data()[i] = fp16_uint16_to_float_nvidia(data_u16->data()[i]);
    }
    return data_f32;
  } else if (dtype.isBF16()) {
    auto data_u16 = read<uint16_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (int i = 0; i < data_u16->size(); i++) {
      unsigned int tmp1 = data_u16->data()[i];
      tmp1=tmp1<<16;
      data_f32->data()[i] = *((float*)&tmp1);
    }
    return data_f32;
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
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  if (dtype.isInteger(8)) {
    return read<uint8_t>();
  } else if (dtype.isF32()) {
    auto data_f32 = read<float>();
    auto bytes = data_f32->size() * sizeof(float);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_f32->data(), bytes);
    return std::move(data_u8);
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    auto bytes = data_i16->size() * sizeof(int16_t);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_i16->data(), bytes);
    return std::move(data_u8);
  } else if (dtype.isInteger(32)) {
    auto data_i32 = read<int32_t>();
    auto bytes = data_i32->size() * sizeof(int32_t);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_i32->data(), bytes);
    return std::move(data_u8);
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
template std::shared_ptr<std::vector<uint16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint8_t>> WeightOp::read();
template std::shared_ptr<std::vector<int32_t>> WeightOp::read();
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<float> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int16_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint16_t> &data,
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
