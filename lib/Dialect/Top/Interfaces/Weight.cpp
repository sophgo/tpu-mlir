//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::top;
using namespace mlir;

template <typename T>
LogicalResult WeightOp::update(const std::vector<T> &data, size_t count) {
  auto op = getOperation();
  auto dialect = op->getDialect();
  auto topDialect = llvm::cast<TopDialect>(dialect);
  if (topDialect->wFile == nullptr) {
    auto moduleOp = Module::getModuleOp(op);
    auto weight_file = Module::getWeightFile(moduleOp);
    topDialect->loadWeightFile(weight_file);
  }
  return topDialect->wFile->updateTensorData(Module::getName(op).str(), &data[0],
                                             count);
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
  auto type = output().getType().cast<RankedTensorType>();
  return topDialect->wFile->readTensor<T>(Module::getName(op).str(), type);
}

std::shared_ptr<std::vector<float>> WeightOp::read_as_float() {
  auto dtype = Module::getStorageType(output());
  if (dtype.isUnsignedInteger(8)) {
    auto data_u8 = read<uint8_t>();
    return std::make_shared<std::vector<float>>(data_u8->begin(),
                                                data_u8->end());
  } else if (dtype.isInteger(8)) {
    auto data_i8 = read<int8_t>();
    return std::make_shared<std::vector<float>>(data_i8->begin(),
                                                data_i8->end());
  } else if (dtype.isF32()) {
    return read<float>();
  } else if (dtype.isF16()) {
    auto data_u16 = read<uint16_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (uint64_t i = 0; i < data_u16->size(); i++) {
      data_f32->data()[i] = f16_to_f32(data_u16->data()[i]);
    }
    return data_f32;
  } else if (dtype.isBF16()) {
    auto data_u16 = read<uint16_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (uint64_t i = 0; i < data_u16->size(); i++) {
      data_f32->data()[i] = bf16_to_f32(data_u16->data()[i]);
    }
    return data_f32;
  } else if (dtype.isUnsignedInteger(16)) {
    auto data_u16 = read<uint16_t>();
    return std::make_shared<std::vector<float>>(data_u16->begin(),
                                                data_u16->end());
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    return std::make_shared<std::vector<float>>(data_i16->begin(),
                                                data_i16->end());
  } else if (dtype.isUnsignedInteger(32)) {
    auto data_u32 = read<uint32_t>();
    return std::make_shared<std::vector<float>>(data_u32->begin(),
                                                data_u32->end());
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
  } else if (dtype.isa<::mlir::Float16Type, ::mlir::BFloat16Type>()) {
    auto data_u16 = read<uint16_t>();
    auto bytes = data_u16->size() * sizeof(uint16_t);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_u16->data(), bytes);
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
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), type);
  return newOp.getResult();
}
template LogicalResult WeightOp::update(const std::vector<uint8_t> &data,
                                        size_t cont);
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
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint32_t> &data,
                                RankedTensorType &type);

mlir::Value WeightOp::clone_bf16(Operation *OwnerOp) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_bf16 = std::make_shared<std::vector<uint16_t>>(count);
  auto is_cv18xx = Module::isCV18xx(Module::getChip(OwnerOp));

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_bf16->at(i) = f32_to_bf16(data->at(i), is_cv18xx, true);
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  auto dialect = ctx->getLoadedDialect("top");
  auto topDialect = llvm::cast<TopDialect>(dialect);
  assert(topDialect->wFile != nullptr);
  std::string new_name = Module::getName(getOperation()).str() + "_bf16";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getBF16Type());
  auto ret =
      topDialect->wFile->addTensor(new_name, data_bf16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type);
  return newOp.getResult();
};

mlir::Value WeightOp::clone_f16(Operation *OwnerOp) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_f16 = std::make_shared<std::vector<uint16_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f16->at(i) = f32_to_f16(data->at(i));
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  auto dialect = ctx->getLoadedDialect("top");
  auto topDialect = llvm::cast<TopDialect>(dialect);
  assert(topDialect->wFile != nullptr);

  std::string new_name = Module::getName(getOperation()).str() + "_f16";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getF16Type());
  auto ret = topDialect->wFile->addTensor(new_name, data_f16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type);
  return newOp.getResult();
};
