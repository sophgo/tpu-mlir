//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

template <typename T>
std::unique_ptr<std::vector<T>>
readTensorFromBytesString(const std::string &byte_str, RankedTensorType &type,
                          uint32_t store_mode, bool do_compress) {
  /// {STORE_MODE_T, align_num}
  std::map<uint32_t, int64_t> stmode_map = {
      {0 /*STORE_MODE_1N*/, 1l},
      {1 /*STORE_MODE_2N*/, 2l},
      {2 /*STORE_MODE_4N*/, 4l},
  };
  size_t count = 1;
  bool isINT4 = false;
  auto s = type.getShape();
  if (s.size() > 0) {
    auto n = type.getShape()[0];
    auto others = type.getNumElements() / n;
    count = (n + stmode_map.at(store_mode) - 1) / stmode_map.at(store_mode) *
            stmode_map.at(store_mode) * others;
    isINT4 = type.getElementType().isInteger(4);
    if (isINT4) {
      auto dims = type.getShape().size();
      if (dims == 2) { // for MatMul
        count = type.getDimSize(0) * ((type.getDimSize(1) + 1) / 2);
      } else if (dims == 4) { // for Conv2d
                              /* count = type.getDimSize(0) *
                                      type.getDimSize(1) *
                                      ((type.getDimSize(2) * type.getDimSize(3) + 1)/2); */
        count = (count + 1) / 2;
      } else {
        assert(0);
      }
    }
  }

  auto data = std::make_unique<std::vector<T>>(count);
  assert(!do_compress);
  int data_size = isINT4 ? 1 : count * sizeof(T);
  std::memcpy(data->data(), byte_str.data(), data_size);
  return data;
}

template <typename T>
std::shared_ptr<std::vector<T>> WeightOp::read() {
  auto op = getOperation();
  auto type = getOutput().getType().cast<RankedTensorType>();
  bool do_compress = getDoCompress().has_value() && getDoCompress().value();
  uint32_t store_mode = 0;
  if (getStoreMode().has_value()) {
    store_mode = llvm::StringSwitch<uint32_t>(getStoreModeAttr())
                     .Case("1N", 0)
                     .Case("2N", 1)
                     .Case("4N", 2)
                     .Default(0);
  }
  if (!getInlineBytes().has_value() || getInlineBytes().value().str().empty()) {
    return module::weightFile().readTensor<T>(module::getName(op).str(), type,
                                              store_mode, do_compress);
  } else {
    return readTensorFromBytesString<T>(getInlineBytes().value().str(), type,
                                        store_mode, do_compress);
  }
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
  } else if (dtype.isFloat8E4M3FN()) {
    auto data_u8 = read<uint8_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u8->size());
    for (uint64_t i = 0; i < data_u8->size(); i++) {
      data_f32->data()[i] = f8e4m3_to_f32(data_u8->data()[i]);
    }
    return data_f32;
  } else if (dtype.isFloat8E5M2()) {
    auto data_u8 = read<uint8_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u8->size());
    for (uint64_t i = 0; i < data_u8->size(); i++) {
      data_f32->data()[i] = f8e5m2_to_f32(data_u8->data()[i]);
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
  } else if (dtype.isa<Float8E4M3FNType, Float8E5M2Type>()) {
    auto data_f8 = read<uint8_t>();
    auto bytes = data_f8->size();
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_f8->data(), bytes);
    return std::move(data_u8);
  }
  dump();
  llvm_unreachable("weight data not support read now");
  return nullptr;
}

std::shared_ptr<std::vector<int8_t>> WeightOp::read_as_f8e4m3() {
  auto dtype = module::getStorageType(getOutput());
  if (!dtype.isFloat8E4M3FN())
    llvm_unreachable("dtype not align");

  auto data_f8 = read<uint8_t>();
  auto data_i8 = std::make_shared<std::vector<int8_t>>();
  for (int i = 0; i < data_f8->size(); i++)
    data_i8->push_back(*((int8_t *)(data_f8->data() + i)));
  return data_i8;
}

std::shared_ptr<std::vector<int8_t>> WeightOp::read_as_f8e5m2() {
  auto dtype = module::getStorageType(getOutput());
  if (!dtype.isFloat8E5M2())
    llvm_unreachable("dtype not align");

  auto data_f8 = read<uint8_t>();

  auto data_i8 = std::make_shared<std::vector<int8_t>>();

  for (int i = 0; i < data_f8->size(); i++)
    data_i8->push_back(*((int8_t *)(data_f8->data() + i)));
  return data_i8;
}

template <typename T>
Value WeightOp::create(Operation *OwnerOp, llvm::StringRef suffix,
                       const std::vector<T> &data, RankedTensorType &type,
                       uint32_t store_mode) {
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  std::string op_name = module::getName(OwnerOp).str();
  std::string new_name = op_name + "_" + suffix.str();
  if (!module::getWeightInMemFlag()) {
    std::set<StringRef> all_tensor_names;
    module::weightFile().getAllNames(all_tensor_names);
    auto it = all_tensor_names.find(new_name.c_str());
    int index = 1;
    while (it != all_tensor_names.end()) {
      new_name = op_name + "_" + std::to_string((index++)) + "_" + suffix.str();
      it = all_tensor_names.find(new_name.c_str());
    }
    auto ret = module::weightFile().addTensor(new_name, &data, type);
    assert(succeeded(ret));
  }
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp =
      builder.create<top::WeightOp>(NameLoc::get(nameAttr), type, ValueRange{});
  auto stmodeAttr = builder.getStringAttr(
      store_mode == 0 ? "1N" : store_mode == 1 ? "2N" : "4N");
  if (stmodeAttr != "1N")
    newOp.setStoreModeAttr(stmodeAttr);
  if (module::getWeightInMemFlag()) {
    std::string inline_bytes((char *)data.data(), data.size() * sizeof(T));
    newOp.setInlineBytesAttr(builder.getStringAttr(inline_bytes));
  }
  return newOp.getResult();
}

template std::shared_ptr<std::vector<float>> WeightOp::read();
template std::shared_ptr<std::vector<int8_t>> WeightOp::read();
template std::shared_ptr<std::vector<int16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint8_t>> WeightOp::read();
template i32_array_t WeightOp::read();
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<float> &data,
                                RankedTensorType &type, uint32_t store_mode);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int16_t> &data,
                                RankedTensorType &type, uint32_t store_mode);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint16_t> &data,
                                RankedTensorType &type, uint32_t store_mode);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int8_t> &data,
                                RankedTensorType &type, uint32_t store_mode);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint8_t> &data,
                                RankedTensorType &type, uint32_t store_mode);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int32_t> &data,
                                RankedTensorType &type, uint32_t store_mode);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint32_t> &data,
                                RankedTensorType &type, uint32_t store_mode);

Value WeightOp::clone_bf16(Operation *OwnerOp, std::string name) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  if (dtype.isBF16())
    return getResult();
  ASSERT_THIS(dtype.isF32());
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
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name =
      name.empty() ? module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_bf16"
                   : name;
  auto new_type = RankedTensorType::get(type.getShape(), builder.getBF16Type());
  if (!module::getWeightInMemFlag()) {
    auto ret =
        module::weightFile().addTensor(new_name, data_bf16->data(), new_type);
    ASSERT_THIS(succeeded(ret));
  }
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  if (module::getWeightInMemFlag()) {
    std::string inline_bytes((char *)data_bf16->data(),
                             data_bf16->size() * sizeof(int16_t));
    newOp.setInlineBytesAttr(builder.getStringAttr(inline_bytes));
  }
  return newOp.getResult();
};

Value WeightOp::clone_f16(Operation *OwnerOp) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  if (dtype.isF16())
    return getResult();
  ASSERT_THIS(dtype.isF32());
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
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_f16";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getF16Type());
  if (!module::getWeightInMemFlag()) {
    auto ret =
        module::weightFile().addTensor(new_name, data_f16->data(), new_type);
    ASSERT_THIS(succeeded(ret));
  }
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  if (module::getWeightInMemFlag()) {
    std::string inline_bytes((char *)data_f16->data(),
                             data_f16->size() * sizeof(int16_t));
    newOp.setInlineBytesAttr(builder.getStringAttr(inline_bytes));
  }
  return newOp.getResult();
};

Value WeightOp::clone_f8e4m3(Operation *OwnerOp, bool per_channel_scale,
                             bool channel_first_dim) {
  auto type = getType().cast<RankedTensorType>();
  auto shape = type.getShape();
  auto dtype = type.getElementType();
  ASSERT_THIS(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_f8 = std::make_shared<std::vector<uint8_t>>(count);
  size_t oc;
  size_t cnt_p_c;
  if (channel_first_dim) {
    oc = shape[0];
    cnt_p_c = count / shape[0];
  } else {
    ASSERT_THIS(shape.size() == 2); // only support 2d matmul
    oc = shape[1];
    cnt_p_c = count / shape[1];
  }

  f64_array_t weight_scale_v;
  if (per_channel_scale) {
    if (getScale().has_value()) {
      weight_scale_v = module::getF64Array(getScale().value());
      ASSERT_THIS(oc == weight_scale_v->size());
    } else {
      // search for the max value and set scale to it
      std::vector<double> weight_scale_v_;
      if (channel_first_dim) {
        for (size_t i = 0; i < oc; i++) {
          float absmax = std::abs(data->at(i * cnt_p_c));
          for (size_t j = 0; j < cnt_p_c; j++) {
            absmax = std::abs(data->at(i * cnt_p_c + j)) > absmax
                         ? std::abs(data->at(i * cnt_p_c + j))
                         : absmax;
          }
          absmax = absmax > 1e-8 ? absmax : 1e-8;
          weight_scale_v_.push_back(absmax / get_f8e4m3_max());
        }
      } else {
        for (size_t i = 0; i < oc; i++) {
          float absmax = std::abs(data->at(oc));
          for (size_t j = 0; j < cnt_p_c; j++) {
            absmax = std::abs(data->at(j * oc + i)) > absmax
                         ? std::abs(data->at(j * oc + i))
                         : absmax;
          }
          absmax = absmax > 1e-8 ? absmax : 1e-8;
          weight_scale_v_.push_back(absmax / get_f8e4m3_max());
        }
      }
      weight_scale_v = std::make_shared<std::vector<double>>(weight_scale_v_);
    }
    if (channel_first_dim) {
#pragma omp parallel for schedule(static, omp_schedule(count))
      for (uint32_t i = 0; i < count; i++) {
        data->at(i) =
            data->at(i) / weight_scale_v.get()->at((int)(i / cnt_p_c));
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(count))
      for (uint32_t i = 0; i < count; i++) {
        data->at(i) = data->at(i) / weight_scale_v.get()->at(i % oc);
      }
    }
  } else {
    float absmax = std::abs(data->at(0));
    for (int i = 0; i < count; i++)
      absmax = absmax > std::abs(data->at(i)) ? absmax : std::abs(data->at(i));
    weight_scale_v =
        std::make_shared<std::vector<double>>(1, absmax / get_f8e4m3_max());
#pragma omp parallel for schedule(static, omp_schedule(count))
    for (uint32_t i = 0; i < count; i++) {
      data->at(i) = data->at(i) / weight_scale_v.get()->at(0);
    }
  }
#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f8->at(i) = f32_to_f8e4m3(data->at(i), true);
  }
  // FIXME: should calculate the scale and set the scale attr
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);

  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_f8e4m3";
  auto new_type =
      RankedTensorType::get(type.getShape(), builder.getFloat8E4M3FNType());
  if (!module::getWeightInMemFlag()) {
    auto ret =
        module::weightFile().addTensor(new_name, data_f8->data(), new_type);
    ASSERT_THIS(succeeded(ret));
  }
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  if (!getScale().has_value()) {
    newOp.getOperation()->setAttr(
        "scale", builder.getF64ArrayAttr(ArrayRef<double>{*weight_scale_v}));
  }
  if (module::getWeightInMemFlag()) {
    std::string inline_bytes((char *)data_f8->data(), data_f8->size());
    newOp.setInlineBytesAttr(builder.getStringAttr(inline_bytes));
  }
  return newOp.getResult();
};

Value WeightOp::clone_f8e5m2(Operation *OwnerOp) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  ASSERT_THIS(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_f8 = std::make_shared<std::vector<uint8_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f8->at(i) = f32_to_f8e5m2(data->at(i), true);
  }
  // FIXME: scale set to 1.0
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_f8e5m2";
  auto new_type = RankedTensorType::get(
      type.getShape(),
      builder.getFloat8E5M2Type()); // builder.getFloat8E5M2Type());
  if (!module::getWeightInMemFlag()) {
    auto ret =
        module::weightFile().addTensor(new_name, data_f8->data(), new_type);
    ASSERT_THIS(succeeded(ret));
  }
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  if (module::getWeightInMemFlag()) {
    std::string inline_bytes((char *)data_f8->data(), data_f8->size());
    newOp.setInlineBytesAttr(builder.getStringAttr(inline_bytes));
  }
  return newOp.getResult();
};

// template <typename Ty>
Value WeightOp::clone_int(Operation *OwnerOp) {
  auto type = getType().cast<RankedTensorType>();
  auto dtype = type.getElementType();
  ASSERT_THIS(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_int = std::make_shared<std::vector<int32_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_int->at(i) = static_cast<int32_t>(data->at(i));
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_int";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getI32Type());
  if (!module::getWeightInMemFlag()) {
    auto ret =
        module::weightFile().addTensor(new_name, data_int->data(), new_type);
    ASSERT_THIS(succeeded(ret));
  }
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  if (module::getWeightInMemFlag()) {
    std::string inline_bytes((char *)data_int->data(),
                             data_int->size() * sizeof(int32_t));
    newOp.setInlineBytesAttr(builder.getStringAttr(inline_bytes));
  }
  return newOp.getResult();
};

Value WeightOp::clone(llvm::StringRef suffix) {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  builder.setInsertionPointAfter(op);
  auto name = module::getName(op);
  auto new_name = name.str() + "_" + suffix.str();
  auto nameAttr = builder.getStringAttr(new_name);
  if (!module::getWeightInMemFlag()) {
    auto ret = module::weightFile().cloneTensor(name, suffix);
    ASSERT_THIS(succeeded(ret));
  }
  auto newOp = builder.create<top::WeightOp>(NameLoc::get(nameAttr), getType(),
                                             ValueRange{});
  if (module::getWeightInMemFlag()) {
    newOp.setInlineBytesAttr(getInlineBytesAttr());
  }
  return newOp.getOutput();
}

template <typename T>
void split_weight(std::shared_ptr<std::vector<T>> &old_data,
                  std::shared_ptr<std::vector<T>> &new_data, int begin, int end,
                  int axis, llvm::ArrayRef<int64_t> &shape) {
  int64_t outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= shape[i];
  }
  int64_t inner = 1;
  for (int i = axis + 1; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  int64_t head_inner = inner * (end - begin);
  inner *= shape[axis];
  for (int64_t i = 0; i < outer; ++i) {
    int64_t src_offset = i * inner + begin * (inner / shape[axis]);
    int64_t dst_offset = i * head_inner;
    for (int64_t j = 0; j < head_inner; ++j) {
      new_data->data()[dst_offset + j] = old_data->at(src_offset + j);
    }
  }
}

Value WeightOp::split(int begin, int end, int axis, mlir::Type to_type,
                      std::string suffix) {
  auto op = getOperation();
  auto shape = module::getShape(getOutput());
  auto dtype = module::getStorageType(getOutput());
  auto dim = shape.size();
  axis = axis < 0 ? dim + axis : axis;

  std::vector<int64_t> out_shape(shape);
  out_shape[axis] = end - begin;
  int64_t out_size =
      module::getNumElements(getOutput()) / shape[axis] * (end - begin);
  auto new_type = RankedTensorType::get(out_shape, to_type);
  if (dtype.isUnsignedInteger(8) || dtype.isFloat8E4M3FN() ||
      dtype.isFloat8E5M2()) {
    auto data = read<uint8_t>();
    auto out_weight = std::make_shared<std::vector<uint8_t>>(out_size);
    split_weight(data, out_weight, begin, end, axis, shape);
    return create(op, suffix, *out_weight, new_type);
  } else if (dtype.isInteger(8)) {
    auto data = read<int8_t>();
    auto out_weight = std::make_shared<std::vector<int8_t>>(out_size);
    split_weight(data, out_weight, begin, end, axis, shape);
    return create(op, suffix, *out_weight, new_type);
  } else if (dtype.isF32()) {
    auto data = read<float>();
    auto out_weight = std::make_shared<std::vector<float>>(out_size);
    split_weight(data, out_weight, begin, end, axis, shape);
    return create(op, suffix, *out_weight, new_type);
  } else if (dtype.isF16() || dtype.isBF16() || dtype.isUnsignedInteger(16)) {
    auto data = read<uint16_t>();
    auto out_weight = std::make_shared<std::vector<uint16_t>>(out_size);
    split_weight(data, out_weight, begin, end, axis, shape);
    return create(op, suffix, *out_weight, new_type);
  } else if (dtype.isInteger(16)) {
    auto data = read<int16_t>();
    auto out_weight = std::make_shared<std::vector<int16_t>>(out_size);
    split_weight(data, out_weight, begin, end, axis, shape);
    return create(op, suffix, *out_weight, new_type);
  } else if (dtype.isUnsignedInteger(32)) {
    auto data = read<uint32_t>();
    auto out_weight = std::make_shared<std::vector<uint32_t>>(out_size);
    split_weight(data, out_weight, begin, end, axis, shape);
    return create(op, suffix, *out_weight, new_type);
  } else if (dtype.isInteger(32)) {
    auto data = read<int32_t>();
    auto out_weight = std::make_shared<std::vector<int32_t>>(out_size);
    split_weight(data, out_weight, begin, end, axis, shape);
    return create(op, suffix, *out_weight, new_type);
  }
  dump();
  llvm_unreachable("weight data not support split now");
  return nullptr;
}

template <typename T>
LogicalResult WeightOp::update(const std::vector<T> &data, size_t count) {
  if (!module::getWeightInMemFlag()) {
    auto op = getOperation();
    return module::weightFile().updateTensorData(module::getName(op).str(),
                                                 &data[0], count);
  } else {
    std::string inline_bytes((char *)data.data(), data.size() * sizeof(T));
    OpBuilder builder(getContext());
    setInlineBytesAttr(builder.getStringAttr(inline_bytes));
    return success();
  }
}

template LogicalResult WeightOp::update(const std::vector<uint8_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<uint16_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<uint32_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<float> &data,
                                        size_t cont);

Value WeightOp::create_float(Operation *OwnerOp, llvm::StringRef suffix,
                             const std::vector<float> &data,
                             const std::vector<int64_t> &shape,
                             Type storage_type) {
  auto f32_type = Float32Type::get(OwnerOp->getContext());
  auto w_type = RankedTensorType::get(shape, f32_type);
  auto weight = WeightOp::create(OwnerOp, suffix, data, w_type);
  if (storage_type.isF16()) {
    auto weight_ =
        dyn_cast<top::WeightOp>(weight.getDefiningOp()).clone_f16(OwnerOp);
    return weight_;
  } else if (storage_type.isBF16()) {
    auto weight_ =
        dyn_cast<top::WeightOp>(weight.getDefiningOp()).clone_bf16(OwnerOp);
    return weight_;
  } else {
    return weight;
  }
}
