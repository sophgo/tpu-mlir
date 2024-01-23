//===- TensorFile.h - utilities for working with tensor files ------------*- C++
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Common utilities for working with tensor files.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TENSORFILE_H_
#define MLIR_SUPPORT_TENSORFILE_H_

#include "cnpy.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <ctime>
#include <fstream>
#include <set>
#include <string>
#include <system_error>
#include <type_traits>

#include <iomanip>

namespace mlir {

class TensorFile {
public:
  TensorFile(llvm::StringRef filename, bool readOnly, bool newCreate = false);

  ~TensorFile();

  /// update a tensor for weight compress
  /// if the name is not found, return failure()
  template <typename T>
  LogicalResult updateTensorData(llvm::StringRef name, const T *data,
                                 size_t count);
  /// add a new tensor to file
  /// if the name is already used, return failure()
  template <typename T>
  LogicalResult addTensor(llvm::StringRef name, const T *data,
                          RankedTensorType &type, int64_t length = 0);

  template <typename T>
  LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
                          RankedTensorType &type);

  /// add a new tensor to file
  /// if the name is already used, return failure()
  template <typename T>
  LogicalResult addTensor(llvm::StringRef name, const T *data,
                          std::vector<int64_t> &shape);

  template <typename T>
  LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
                          std::vector<int64_t> &shape);

  LogicalResult cloneTensor(llvm::StringRef name, llvm::StringRef suffix);

  /// read a tensor from file
  /// if the name is not found, return failure()
  /// type is provided for checking, return failure() if type does not match
  template <typename T>
  LogicalResult readTensor(llvm::StringRef name, T *data, size_t count,
                           bool isINT4, bool do_compress);
  template <typename T>
  std::unique_ptr<std::vector<T>>
  readTensor(llvm::StringRef name, RankedTensorType &type, uint32_t store_mode, bool do_compress = false);

  /// delete a tensor from file
  /// if the name is not found, return failure()
  LogicalResult deleteTensor(const llvm::StringRef name);

  void getAllNames(std::set<StringRef> &names);

  /// read all tensor from file
  template <typename T>
  LogicalResult readAllTensors(std::vector<std::string> &names,
                               std::vector<std::vector<T> *> &tensors,
                               std::vector<std::vector<int64_t>> &shapes);

  bool changed();

  template <typename T>
  void colMajorToRowMajor(T &des, const cnpy::NpyArray &src);
  void save(const std::string &file = "");

private:
  /// load the file
  LogicalResult load(void);

  std::string filename;
  bool readOnly;
  cnpy::npz_t map;
  std::atomic<int> cnt_del = {0};
  std::atomic<int> cnt_add = {0};
  std::atomic<int> cnt_update = {0};
};

} // namespace mlir

#endif // MLIR_SUPPORT_TENSORFILE_H_
