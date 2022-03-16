
//===- FileUtilities.h - utilities for working with tensor files -------*- C++
//-*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Common utilities for working with tensor files.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TENSORFILE_H_
#define MLIR_SUPPORT_TENSORFILE_H_

#include "cnpy.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"

#include <type_traits>
#include <string>
#include <system_error>
#include <ctime>
#include <atomic>
#include <fstream>
#include <set>

#include <iomanip>
template <typename T> static std::string int_to_hex(T i) {
  std::stringstream stream;
  stream << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex << i;
  return stream.str();
}

namespace mlir {

template <typename T> static bool check_type(Type eltType) {
  bool same;
  if (eltType.isBF16()) {
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
    same = std::is_same<T, uint32_t>::value;
  } else {
    // eltType.isF16()
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

class TensorFile {
public:
  TensorFile(llvm::StringRef filename, bool readOnly, bool newCreate = false)
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
          assert(0);
        }
        map.clear();
      }
    } else {
      map.clear();
    }
  }

  ~TensorFile() {}

  /// add a new tensor to file
  /// if the name is already used, return failure()
  template <typename T>
  LogicalResult addTensor(llvm::StringRef name, const T *data,
                          RankedTensorType &type) {
    assert(!readOnly);
    assert(check_type<T>(type.getElementType()) == true);
    auto it = map.find(name.str());
    if (it != map.end()) {
      llvm::errs() << "failed to add tensor " << name.str()
                   << ", already exist\n";
      return failure();
    }
    std::vector<int64_t> shape = type.getShape();
    std::vector<size_t> shape_npz;
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      shape_npz.push_back((size_t)*it);
    }
    cnpy::npz_add_array(map, name.str(), &data[0], shape_npz);
    cnt_add++;
    return success();
  }

  template <typename T>
  LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
                          RankedTensorType &type) {
    assert(!readOnly);
    assert(check_type<T>(type.getElementType()) == true);
    return addTensor(name, data->data(), type);
  }

  /// add a new tensor to file
  /// if the name is already used, return failure()
  template <typename T>
  LogicalResult addTensor(llvm::StringRef name, const T *data,
                          std::vector<int64_t> &shape) {
    assert(!readOnly);
    auto it = map.find(name.str());
    if (it != map.end()) {
      llvm::errs() << "failed to add tensor " << name.str()
                   << ", already exist\n";
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
  LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
                          std::vector<int64_t> &shape) {
    assert(!readOnly);
    return addTensor(name, data->data(), shape);
  }

  /// read a tensor from file
  /// if the name is not found, return failure()
  /// type is provided for checking, return failure() if type does not match
  template <typename T>
  LogicalResult readTensor(llvm::StringRef name, T *data, size_t count) {
    auto it = map.find(name.str());
    if (it == map.end()) {
      llvm::errs() << "failed to find tensor " << name.str() << " to read\n";
      return failure();
    }
    auto arr = it->second;
    if (arr.num_bytes() != count * sizeof(T)) {
      llvm::errs() << "size does not match for tensor " << name.str() << "\n";
      return failure();
    }
    memcpy(data, arr.data_holder->data(), arr.num_bytes());
    return success();
  }

  template <typename T>
  std::unique_ptr<std::vector<T>> readTensor(llvm::StringRef name,
                                             RankedTensorType &type) {
    assert(check_type<T>(type.getElementType()) == true);
    std::vector<int64_t> shape = type.getShape();
    auto count = std::accumulate(std::begin(shape), std::end(shape), 1,
                                 std::multiplies<>());
    auto data = std::make_unique<std::vector<T>>(count);
    auto ret = readTensor(name, (T *)data.get()->data(), count);
    assert(succeeded(ret));
    return data;
  }

  /// delete a tensor from file
  /// if the name is not found, return failure()
  LogicalResult deleteTensor(const llvm::StringRef name) {
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

  void getAllNames(std::set<StringRef> &names) {
    for (auto &name : map) {
      names.insert(name.first);
    }
  }

  /// read all tensor from file
  template <typename T>
  LogicalResult readAllTensors(std::vector<std::string> &names,
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
      assert(count == (size_t)std::accumulate(std::begin(shape),
                                              std::end(shape), 1,
                                              std::multiplies<>()));
      shapes.push_back(shape);
      names.push_back(it->first);
    }
    return success();
  }

  bool changed() { return cnt_add + cnt_del > 0; }

  void save(const std::string &file = "") {
    assert(!readOnly);
    if (cnt_add + cnt_del == 0) {
      return;
    }
    if (!file.empty()) {
      filename = file;
    }

    cnpy::npz_save_all(filename, map);

    int ret = cnt_add + cnt_del;
    cnt_add = 0;
    cnt_del = 0;
    return;
  }

private:
  /// load the file
  LogicalResult load(void) {
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
};

/// Open the file specified by its name for reading.
std::unique_ptr<TensorFile> openInputTensorFile(llvm::StringRef filename);

/// Create a new file specified by its name for writing.
std::unique_ptr<TensorFile>
openOutputTensorFile(llvm::StringRef outputFilename);

/// Open a existing file specified by its name for updating, create one if not
/// exist.
std::unique_ptr<TensorFile> openTensorFile(llvm::StringRef filename);

} // namespace mlir

#endif // MLIR_SUPPORT_TENSORFILE_H_
