//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/TensorFile.h"

namespace mlir {

template <typename T>
static std::string int_to_hex(T i) {
  std::stringstream stream;
  stream << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex << i;
  return stream.str();
}

template <typename T>
static bool check_type(Type eltType) {
  if (eltType.isa<quant::UniformQuantizedPerAxisType>()) {
    eltType =
        eltType.cast<quant::UniformQuantizedPerAxisType>().getStorageType();
  } else if (eltType.isa<quant::UniformQuantizedType>()) {
    eltType = eltType.cast<quant::UniformQuantizedType>().getStorageType();
  }

  bool same;
  if (eltType.isBF16() || eltType.isF16()) {
    // we use uint16_t to represent BF16 (same as tensorflow)
    same = std::is_same<T, uint16_t>::value;
  } else if (eltType.isF32()) {
    same = std::is_same<T, float>::value;
  } else if (eltType.isInteger(8)) {
    same = (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value);
  } else if (eltType.isInteger(16)) {
    same =
        (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value);
  } else if (eltType.isInteger(32)) {
    same =
        (std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value);
  } else if (eltType.isInteger(4)) {
    same = (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value);
  } else if (eltType.isFloat8E4M3FN()) {
    same = std::is_same<T, uint8_t>::value;
  } else if (eltType.isFloat8E5M2()) {
    same = std::is_same<T, uint8_t>::value;
  } else {
    // eltType.isF64()
    // ...
    same = false;
  }
  if (same != true) {
    eltType.dump();
    llvm::errs() << "\nnot equal to Type "
                 << "\n";
  }
  return same;
}

TensorFile::TensorFile(llvm::StringRef filename, bool readOnly, bool newCreate)
    : filename(filename), readOnly(readOnly) {
  if (!newCreate) {
    std::ifstream f(filename.str());
    if (!f.good()) {
      llvm::errs() << "WARNING, " << filename
                   << " doesn't exist, please check\n";
    }
    auto ret = load();
    if (!succeeded(ret)) {
      if (readOnly) {
        llvm::errs() << filename << " not exist, failed to read for read\n";
        llvm_unreachable("TensorFile error!");
      }
      map.clear();
    }
  } else {
    map.clear();
  }
}

TensorFile::~TensorFile() {}

/// update a tensor for weight compress
/// if the name is not found, return failure()
template <typename T>
LogicalResult TensorFile::updateTensorData(llvm::StringRef name, const T *data,
                                           size_t count) {
  assert(!readOnly);
  auto it = map.find(name.str());
  if (it == map.end()) {
    llvm::errs() << "failed to add tensor " << name.str()
                 << ", already exist\n";
    llvm_unreachable("addTensor error!");
    return failure();
  }
  cnpy::NpyArray &arr = it->second;
  if (arr.num_bytes() != count * sizeof(T)) {
    llvm::errs() << "size does not match for tensor " << name.str() << " "
                 << count * sizeof(T) << " vs " << arr.num_bytes() << "\n";
    llvm_unreachable("readTensor failed");
    return failure();
  }
  arr.fortran_order = false;
  memcpy(arr.data_holder->data(), data, arr.num_bytes());
  cnt_update++;
  return success();
}

LogicalResult TensorFile::cloneTensor(llvm::StringRef name,
                                      llvm::StringRef suffix) {
  assert(!readOnly);
  auto it = map.find(name.str());
  if (it == map.end()) {
    llvm::errs() << "failed to clone tensor " << name.str() << ", not exist\n";
    llvm_unreachable("cloneTensor error!");
    return failure();
  }
  auto new_name = name.str() + "_" + suffix.str();
  auto new_it = map.find(new_name);
  if (new_it != map.end()) {
    llvm::errs() << "failed to clone tensor " << new_name << ", exist\n";
    llvm_unreachable("cloneTensor error!");
    return failure();
  }
  cnpy::npz_clone_array(map, name.str(), new_name);
  return success();
}

/// add a new tensor to file
/// if the name is already used, return failure()
template <typename T>
LogicalResult TensorFile::addTensor(llvm::StringRef name, const T *data,
                                    RankedTensorType &type, int64_t length) {
  assert(!readOnly);
  assert(check_type<T>(type.getElementType()) == true);
  auto it = map.find(name.str());
  if (it != map.end()) {
    llvm::errs() << "failed to add tensor " << name.str()
                 << ", already exist\n";
    llvm_unreachable("addTensor error!");
    return failure();
  }
  std::vector<int64_t> shape = type.getShape();
  std::vector<size_t> shape_npz;
  if (type.getElementType().isInteger(4) || length != 0) {
    shape_npz.push_back((size_t)length);
  } else {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      shape_npz.push_back((size_t)*it);
    }
  }
  cnpy::npz_add_array(map, name.str(), &data[0], shape_npz);
  cnt_add++;
  return success();
}

template <typename T>
LogicalResult TensorFile::addTensor(llvm::StringRef name,
                                    const std::vector<T> *data,
                                    RankedTensorType &type) {
  assert(!readOnly);
  assert(check_type<T>(type.getElementType()) == true);
  return addTensor(name, data->data(), type, data->size());
}

/// add a new tensor to file
/// if the name is already used, return failure()
template <typename T>
LogicalResult TensorFile::addTensor(llvm::StringRef name, const T *data,
                                    std::vector<int64_t> &shape) {
  assert(!readOnly);
  auto it = map.find(name.str());
  if (it != map.end()) {
    llvm::errs() << "failed to add tensor " << name.str()
                 << ", already exist\n";
    llvm_unreachable("addTensor error!");
    return failure();
  }
  std::vector<size_t> shape_npz;
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    shape_npz.push_back((size_t)*it);
  }
  cnpy::npz_add_array(map, name.str(), &data[0], shape_npz);
  cnt_add++;
  return success();
}

template <typename T>
LogicalResult TensorFile::addTensor(llvm::StringRef name,
                                    const std::vector<T> *data,
                                    std::vector<int64_t> &shape) {
  assert(!readOnly);
  return addTensor(name, data->data(), shape);
}

template LogicalResult
TensorFile::addTensor<unsigned short>(llvm::StringRef name,
                                      const unsigned short *data,
                                      RankedTensorType &type, int64_t length);
template LogicalResult
TensorFile::addTensor<signed char>(llvm::StringRef name,
                                   const signed char *data,
                                   RankedTensorType &type, int64_t length);
template LogicalResult
TensorFile::addTensor<signed char>(llvm::StringRef name,
                                   const std::vector<signed char> *data,
                                   RankedTensorType &type);
template LogicalResult
TensorFile::addTensor<unsigned char>(llvm::StringRef name,
                                     const std::vector<unsigned char> *data,
                                     RankedTensorType &type);
template LogicalResult
TensorFile::addTensor<short>(llvm::StringRef name,
                             const std::vector<short> *data,
                             RankedTensorType &type);
template LogicalResult
TensorFile::addTensor<float>(llvm::StringRef name,
                             const std::vector<float> *data,
                             RankedTensorType &type);

template LogicalResult TensorFile::updateTensorData<unsigned char>(
    llvm::StringRef name, const unsigned char *data, size_t count);
template LogicalResult TensorFile::updateTensorData<float>(llvm::StringRef name,
                                                           const float *data,
                                                           size_t count);
template LogicalResult TensorFile::updateTensorData<unsigned int>(
    llvm::StringRef name, const unsigned int *data, size_t count);
template LogicalResult
TensorFile::addTensor<unsigned short>(llvm::StringRef name,
                                      const std::vector<unsigned short> *data,
                                      RankedTensorType &type);
template LogicalResult
TensorFile::addTensor<unsigned int>(llvm::StringRef name,
                                    const std::vector<unsigned int> *data,
                                    RankedTensorType &type);
template LogicalResult TensorFile::addTensor<int>(llvm::StringRef name,
                                                  const std::vector<int> *data,
                                                  RankedTensorType &type);

template LogicalResult TensorFile::updateTensorData(llvm::StringRef name,
                                                    const unsigned short *data,
                                                    size_t count);

template LogicalResult
TensorFile::readAllTensors(std::vector<std::string> &names,
                           std::vector<std::vector<int> *> &tensors,
                           std::vector<std::vector<int64_t>> &shapes);

template std::unique_ptr<std::vector<int>>
TensorFile::readTensor<int>(llvm::StringRef name, RankedTensorType &type,
                            uint32_t store_mode, bool do_compress);
template std::unique_ptr<std::vector<unsigned int>>
TensorFile::readTensor<unsigned int>(llvm::StringRef name,
                                     RankedTensorType &type,
                                     uint32_t store_mode, bool do_compress);
template std::unique_ptr<std::vector<short>>
TensorFile::readTensor<short>(llvm::StringRef name, RankedTensorType &type,
                              uint32_t store_mode, bool do_compress);
template std::unique_ptr<std::vector<float>>
TensorFile::readTensor<float>(llvm::StringRef name, RankedTensorType &type,
                              uint32_t store_mode, bool do_compress);
template std::unique_ptr<std::vector<unsigned char>>
TensorFile::readTensor<unsigned char>(llvm::StringRef name,
                                      RankedTensorType &type,
                                      uint32_t store_mode, bool do_compress);
template std::unique_ptr<std::vector<signed char>>
TensorFile::readTensor<signed char>(llvm::StringRef name,
                                    RankedTensorType &type, uint32_t store_mode,
                                    bool do_compress);
template std::unique_ptr<std::vector<unsigned short>>
TensorFile::readTensor<unsigned short>(llvm::StringRef name,
                                       RankedTensorType &type,
                                       uint32_t store_mode, bool do_compress);
/// read a tensor from file
/// if the name is not found, return failure()
/// type is provided for checking, return failure() if type does not match
template <typename T>
LogicalResult TensorFile::readTensor(llvm::StringRef name, T *data,
                                     size_t count, bool isINT4,
                                     bool do_compress) {
  auto it = map.find(name.str());
  if (it == map.end()) {
    llvm::errs() << "failed to find tensor " << name.str() << " to read\n";
    llvm_unreachable("readTensor failed");
    return failure();
  }
  auto arr = it->second;
  if (arr.num_bytes() != count * sizeof(T) && !isINT4 && !do_compress) {
    llvm::errs() << "size does not match for tensor " << name.str() << "\n";
    llvm_unreachable("readTensor failed");
    return failure();
  }
  if (arr.fortran_order) {
    llvm::MutableArrayRef<char> data_holder((char *)data,
                                            (char *)(data + arr.num_vals));
    colMajorToRowMajor(data_holder, arr);
  } else {
    if (isINT4)
      memcpy(data, arr.data_holder->data(), count);
    else
      memcpy(data, arr.data_holder->data(), arr.num_bytes());
  }
  return success();
}

template <typename T>
std::unique_ptr<std::vector<T>>
TensorFile::readTensor(llvm::StringRef name, RankedTensorType &type,
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
    assert(check_type<T>(type.getElementType()) == true);
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
  auto ret =
      readTensor(name, (T *)data.get()->data(), count, isINT4, do_compress);
  assert(succeeded(ret));
  return data;
}

/// delete a tensor from file
/// if the name is not found, return failure()
LogicalResult TensorFile::deleteTensor(const llvm::StringRef name) {
  assert(!readOnly);
  if (readOnly)
    return failure();
  auto it = map.find(name.str());
  if (it == map.end()) {
    llvm::errs() << "failed to find tensor " << name.str() << " to delete\n";
    return failure();
  }
  map.erase(it);
  cnt_del++;
  return success();
}

void TensorFile::getAllNames(std::set<StringRef> &names) {
  for (auto &name : map) {
    names.insert(name.first);
  }
}

/// read all tensor from file
template <typename T>
LogicalResult
TensorFile::readAllTensors(std::vector<std::string> &names,
                           std::vector<std::vector<T> *> &tensors,
                           std::vector<std::vector<int64_t>> &shapes) {
  for (auto it = map.begin(); it != map.end(); it++) {
    auto arr = it->second;
    assert(arr.type == 'f'); // support float only for now
    assert(arr.word_size == sizeof(float));
    auto count = arr.num_bytes() / arr.word_size;
    std::vector<T> *tensor = new std::vector<T>(count);
    memcpy(tensor->data(), arr.data_holder->data(), arr.num_bytes());
    tensors.push_back(tensor);
    std::vector<int64_t> shape(arr.shape.size());
    shape.assign(arr.shape.begin(), arr.shape.end());
    assert(count == (size_t)std::accumulate(std::begin(shape), std::end(shape),
                                            1, std::multiplies<>()));
    shapes.push_back(shape);
    names.push_back(it->first);
  }
  return success();
}

bool TensorFile::changed() { return cnt_update + cnt_add + cnt_del > 0; }

template <typename T>
void TensorFile::colMajorToRowMajor(T &des, const cnpy::NpyArray &src) {
  static_assert(std::is_same<typename T::value_type, char>::value,
                "container value should be char");
  assert(des.size() == src.num_bytes());
  size_t word_size = src.word_size;
  for (size_t src_offset = 0, src_size = src.num_vals; src_offset < src_size;
       ++src_offset) {
    size_t des_offset = 0;
    size_t ind_n /*ind(0)*/ = src_offset, sub_n /*sub(0)*/ = 0;
    for (auto n : src.shape) {
      sub_n = ind_n % n; //  sub(n) = ind(n) % n
      des_offset = des_offset * n + sub_n;
      ind_n = (ind_n - sub_n) / n; // ind(n+1) = (ind(n) - sub(n)) / n
    }
    memcpy(des.data() + des_offset * word_size,
           src.data_holder->data() + src_offset * word_size, word_size);
  }
}

void TensorFile::save(const std::string &file) {
  assert(!readOnly);
  bool same_name = true;
  if (!file.empty() && file != filename) {
    same_name = false;
    filename = file;
  }
  if (cnt_add + cnt_del + cnt_update == 0 && same_name) {
    return;
  }
  for (auto &it : map) {
    cnpy::NpyArray &array = it.second;
    if (array.fortran_order == true) {
      auto data_holder = std::shared_ptr<std::vector<char>>(
          new std::vector<char>(array.num_bytes()));
      colMajorToRowMajor(*data_holder.get(), array);
      array.data_holder = data_holder;
      array.fortran_order = false;
    }
  }

  cnpy::npz_save_all(filename, map);
  cnt_add = 0;
  cnt_del = 0;
  cnt_update = 0;
  return;
}

LogicalResult TensorFile::load(void) {
  map = cnpy::npz_load(filename);
  if (map.size() > 0) {
    return success();
  } else {
    return failure();
  }
}

std::string filename;
bool readOnly;
cnpy::npz_t map;
std::atomic<int> cnt_del = {0};
std::atomic<int> cnt_add = {0};
std::atomic<int> cnt_update = {0};
}; // namespace mlir
