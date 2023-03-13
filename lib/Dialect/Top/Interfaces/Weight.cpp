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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

template <typename T>
LogicalResult WeightOp::update(const std::vector<T> &data, size_t count) {
  auto op = getOperation();
  auto dialect = op->getDialect();
  auto topDialect = llvm::cast<TopDialect>(dialect);
  if (topDialect->wFile == nullptr) {
    auto weight_file = module::getWeightFile();
    topDialect->loadWeightFile(weight_file);
  }
  return topDialect->wFile->updateTensorData(module::getName(op).str(),
                                             &data[0], count);
}

template <typename T> std::shared_ptr<std::vector<T>> WeightOp::read() {
  auto op = getOperation();
  auto dialect = op->getDialect();
  auto topDialect = llvm::cast<TopDialect>(dialect);
  if (topDialect->wFile == nullptr) {
    auto weight_file = module::getWeightFile();
    topDialect->loadWeightFile(weight_file);
  }
  auto type = getOutput().getType().cast<RankedTensorType>();
  return topDialect->wFile->readTensor<T>(module::getName(op).str(), type);
}

std::shared_ptr<std::vector<float>> WeightOp::read_as_float() {
  auto dtype = module::getStorageType(getOutput());
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
  llvm_unreachable("weight data not support read as float now");
  return nullptr;
}

std::shared_ptr<std::vector<int32_t>> WeightOp::read_as_int32() {
  auto dtype = module::getStorageType(getOutput());
  if (dtype.isInteger(32)) {
    return read<int32_t>();
  } else if (dtype.isUnsignedInteger(16)) {
    auto data_u16 = read<uint16_t>();
    return std::make_shared<std::vector<int32_t>>(data_u16->begin(),
                                                  data_u16->end());
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    return std::make_shared<std::vector<int32_t>>(data_i16->begin(),
                                                  data_i16->end());
  } else if (dtype.isUnsignedInteger(8)) {
    auto data_u8 = read<uint8_t>();
    return std::make_shared<std::vector<int32_t>>(data_u8->begin(),
                                                  data_u8->end());
  } else if (dtype.isInteger(8)) {
    auto data_i8 = read<int8_t>();
    return std::make_shared<std::vector<int32_t>>(data_i8->begin(),
                                                  data_i8->end());
  }
  dump();
  llvm_unreachable("weight data not support read as int32 now");
  return nullptr;
}

std::shared_ptr<std::vector<uint8_t>> WeightOp::read_as_byte() {
  auto dtype = module::getStorageType(getOutput());
  if (dtype.isInteger(8) || dtype.isInteger(4)) {
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
  } else if (dtype.isa<Float16Type, BFloat16Type>()) {
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
    auto weight_file = module::getWeightFile();
    topDialect->loadWeightFile(weight_file);
  }
  std::string op_name = module::getName(OwnerOp).str();
  std::string new_name = op_name + "_" + suffix.str();
  std::set<StringRef> all_tensor_names;
  topDialect->wFile->getAllNames(all_tensor_names);
  auto it = all_tensor_names.find(new_name.c_str());
  int index = 1;
  while (it != all_tensor_names.end()) {
    new_name = op_name + "_" + std::to_string((index++)) + "_" + suffix.str();
    it = all_tensor_names.find(new_name.c_str());
  }

  auto ret = topDialect->wFile->addTensor(new_name, &data, type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp =
      builder.create<top::WeightOp>(NameLoc::get(nameAttr), type, ValueRange{});
  return newOp.getResult();
}
template LogicalResult WeightOp::update(const std::vector<uint8_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<uint16_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<uint32_t> &data,
                                        size_t cont);
template std::shared_ptr<std::vector<float>> WeightOp::read();
template std::shared_ptr<std::vector<int8_t>> WeightOp::read();
template std::shared_ptr<std::vector<int16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint8_t>> WeightOp::read();
template i32_array_t WeightOp::read();
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

Value WeightOp::clone_bf16(Operation *OwnerOp) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_bf16 = std::make_shared<std::vector<uint16_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_bf16->at(i) = f32_to_bf16(data->at(i));
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  auto dialect = ctx->getLoadedDialect("top");
  auto topDialect = llvm::cast<TopDialect>(dialect);
  assert(topDialect->wFile != nullptr);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_bf16";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getBF16Type());
  auto ret =
      topDialect->wFile->addTensor(new_name, data_bf16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  return newOp.getResult();
};

Value WeightOp::clone_f16(Operation *OwnerOp) {
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
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_f16";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getF16Type());
  auto ret = topDialect->wFile->addTensor(new_name, data_f16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  return newOp.getResult();
};

// template <typename Ty>
Value WeightOp::clone_int(Operation *OwnerOp) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_f16 = std::make_shared<std::vector<int32_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f16->at(i) = static_cast<int32_t>(data->at(i));
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  auto dialect = ctx->getLoadedDialect("top");
  auto topDialect = llvm::cast<TopDialect>(dialect);
  assert(topDialect->wFile != nullptr);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_int";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getI32Type());
  auto ret = topDialect->wFile->addTensor(new_name, data_f16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  return newOp.getResult();
};
