
//===- FileUtilities.h - utilities for working with tensor files -------*- C++ -*-===//
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

#include <iomanip>
template<typename T >
static std::string int_to_hex( T i ) {
  std::stringstream stream;
  stream << std::setfill ('0') << std::setw(sizeof(T)*2)
         << std::hex << i;
  return stream.str();
}

namespace mlir {

template< typename T>
static bool check_type(Type eltType) {
  bool same;
  if ( eltType.isBF16() ) {
    // we use uint16_t to represent BF16 (same as tensorflow)
    same = std::is_same<T, uint16_t>::value;
  } else if ( eltType.isF32() ) {
    same = std::is_same<T, float>::value;
  } else if ( eltType.isInteger(8) ) {
    same = (std::is_same<T, int8_t>::value
            || std::is_same<T, uint8_t>::value);
  } else if ( eltType.isInteger(16) ) {
    same = (std::is_same<T, int16_t>::value
            || std::is_same<T, uint16_t>::value);
  } else if ( eltType.isInteger(32) ) {
    same = std::is_same<T, uint32_t>::value;
  } else {
    // eltType.isF16()
    // eltType.isF64()
    // ...
    same = false;
  }
  if (same != true) {
    eltType.dump();
    llvm::errs() << "\nnot equal to Type " << "\n";
  }
  return same;
}

class TensorFile {
public:
  TensorFile(llvm::StringRef filename, std::error_code &EC, bool readOnly,
      bool newCreate = false)
      : filename(filename), readOnly(readOnly) {
    if (!newCreate) {
      std::ifstream f(filename.str());
      if (!f.good()) {
        llvm::errs() << "WARNING, " << filename << " doesn't exist, please check\n";
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

  static std::string generateName(llvm::StringRef base, int index) {
    srand(time(0));
    uint32_t unique = (uint32_t)random();
    uint32_t pid = getpid();
    std::string name = base.str()
                       + "_" + std::to_string(index)
                       + "_" + int_to_hex<uint16_t>(pid)
                       + int_to_hex<uint32_t>(unique)
                       + ".npz";
    return name;
  }

  static std::string incrementName(llvm::StringRef name) {
    auto stem = llvm::sys::path::stem(name);
    SmallVector<StringRef, 100> ss;
    stem.split(ss, '_');
    std::string base;
    for (unsigned i = 0, e = ss.size(); i < e; ++i) {
      if (i < e - 2)
        base += ss[i];
    }
    unsigned long long index = 0;
    ss[ss.size() - 2].consumeInteger(0, index);
    //llvm::errs() << " index : " << index << "\n";
    //llvm::errs() << " base  : " << base << "\n";
    return generateName(base, index + 1);
  }

  /// add a new tensor to file
  /// if the name is already used, return failure()
  template<typename T>
  LogicalResult addTensor(llvm::StringRef name, const T* data,
      TensorType &type) {
    assert(!readOnly);
    assert(check_type<T>(type.getElementType()) == true);
    auto it = map.find(name.str());
    if (it != map.end()) {
      llvm::errs() << "failed to add tensor " << name.str() << ", already exist\n";
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

  template<typename T>
  LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
      TensorType &type) {
    assert(!readOnly);
    assert(check_type<T>(type.getElementType()) == true);
    return addTensor(name, data->data(), type);
  }

  /// add a new tensor to file
  /// if the name is already used, return failure()
  template<typename T>
  LogicalResult addTensor(llvm::StringRef name, const T* data,
      std::vector<int64_t> &shape) {
    assert(!readOnly);
    auto it = map.find(name.str());
    if (it != map.end()) {
      llvm::errs() << "failed to add tensor " << name.str() << ", already exist\n";
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

  template<typename T>
  LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
      std::vector<int64_t> &shape) {
    assert(!readOnly);
    return addTensor(name, data->data(), shape);
  }

  /// read a tensor from file
  /// if the name is not found, return failure()
  /// type is provided for checking, return failure() if type does not match
  template<typename T>
  LogicalResult readTensor(llvm::StringRef name, T* data, size_t count) {
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

  template<typename T>
  std::unique_ptr<std::vector<T> > readTensor(llvm::StringRef name,
      TensorType &type) {
    assert(check_type<T>(type.getElementType()) == true);
    std::vector<int64_t> shape = type.getShape();
    auto count = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto data = std::make_unique<std::vector<T> >(count);
    auto ret = readTensor(name, (T*)data.get()->data(), count);
    assert(succeeded(ret));
    return data;
  }

  /// delete a tensor from file
  /// if the name is not found, return failure()
  template<typename T>
  LogicalResult deleteTensor(llvm::StringRef name) {
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

  /// read all tensor from file
  template<typename T>
  LogicalResult readAllTensors(
      std::vector<std::string> &names,
      std::vector<std::vector<T> *> &tensors,
      std::vector<std::vector<int64_t> > &shapes) {
    for (auto it = map.begin(); it != map.end(); it++) {
      auto arr = it->second;
      assert (arr.type == 'f'); // support float only for now
      assert(arr.word_size == sizeof(float));
      auto count = arr.num_bytes() / arr.word_size;
      std::vector<T> *tensor = new std::vector<T>(count);
      memcpy(tensor->data(), arr.data_holder->data(), arr.num_bytes());
      tensors.push_back(tensor);
      std::vector<int64_t> shape(arr.shape.size());
      shape.assign(arr.shape.begin(), arr.shape.end());
      assert(count == (size_t)std::accumulate(std::begin(shape),
        std::end(shape), 1, std::multiplies<>()));
      shapes.push_back(shape);
      names.push_back(it->first);
    }
    return success();
  }

  int keep(bool incIndex = false, std::string *newName = nullptr) {
    assert(!readOnly);
    if (cnt_add + cnt_del == 0) {
      if (newName) {
        *newName = filename;
      }
      return 0;
    }
    if (incIndex) {
      auto fileInc = TensorFile::incrementName(filename);
      auto first_element = map.begin();
      cnpy::NpyArray &arr = first_element->second;
      if (arr.shape.size() == 0) {
        // first should be create somthing cuz cnpy save flow
        llvm::StringRef name = first_element->first;
        llvm::errs()
            << name
            << "save dummy for prevent open npz file under append mode fail\n";
        (void)deleteTensor<float>(name);
        std::vector<float> fake_data(1);
        std::vector<int64_t> shape(1, 1);
        (void)addTensor(name, fake_data.data(), shape);
      }

      cnpy::npz_save_all(fileInc, map);
      filename = fileInc;
      if (newName) {
        *newName = filename;
      }
    } else {
      cnpy::npz_save_all(filename, map);
    }
    int ret = cnt_add + cnt_del;
    cnt_add = 0;
    cnt_del = 0;
    return ret;
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

/// Open the file specified by its name for reading. Write the error message to
/// `errorMessage` if errors occur and `errorMessage` is not nullptr.
std::unique_ptr<TensorFile>
openInputTensorFile(llvm::StringRef filename,
              std::string *errorMessage = nullptr);

/// Create a new file specified by its name for writing. Write the error message to
/// `errorMessage` if errors occur and `errorMessage` is not nullptr.
std::unique_ptr<TensorFile>
openOutputTensorFile(llvm::StringRef outputFilename,
               std::string *errorMessage = nullptr);

/// Open a existing file specified by its name for updating, create one if not
/// exist. Write the error message to `errorMessage` if errors occur and
/// `errorMessage` is not nullptr.
std::unique_ptr<TensorFile>
openTensorFile(llvm::StringRef filename,
               std::string *errorMessage = nullptr);

} // namespace mlir

#endif // MLIR_SUPPORT_TENSORFILE_H_
