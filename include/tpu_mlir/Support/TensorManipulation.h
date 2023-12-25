//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <variant>

#ifndef TPUMLIR_SUPPORT_TENSORMANIPULATION_H_
#define TPUMLIR_SUPPORT_TENSORMANIPULATION_H_

namespace tpu_mlir {
using namespace llvm;

template <typename DType>
class Tensor {
  SmallVector<int64_t> strides;
  SmallVector<int64_t> shape;
  std::vector<DType> data;

public:
  Tensor() = default;

  template <typename T>
  Tensor(const std::vector<DType> &source, std::initializer_list<T> _shape) {
    for (auto rank : _shape) {
      shape.push_back(rank);
    }
    setStrides();
    data = source;
  }

  template <typename T>
  Tensor(const std::vector<DType> &source, ArrayRef<T> _shape) {
    for (auto rank : _shape) {
      shape.push_back(rank);
    }
    setStrides();
    data = source;
  }

  std::vector<DType> &getData() { return data; }
  SmallVector<int64_t> &getShape() { return shape; }
  SmallVector<int64_t> &getStrides() { return strides; }

  DType &operator[](size_t n) { return data[n]; }

  void setStrides() {
    auto dims = shape.size();
    strides.resize(dims);
    for (int64_t i = dims - 1, stride = 1; i >= 0; i--) {
      auto rank = shape[i];
      strides[i] = stride;
      stride *= rank;
    }
  }
  template <typename T>
  void setStrides(SmallVector<T> _strides) {
    strides.assign(_strides);
  }

  template <typename T>
  void reshape(SmallVector<T> _shape) {
    size_t size = 1;
    shape.clear();
    for (auto rank : _shape) {
      shape.push_back(rank);
      size *= rank;
    }
    setStrides();
    data.resize(size);
  }

  template <typename T>
  Tensor(const std::vector<DType> &source, std::initializer_list<T> _shape,
         std::initializer_list<T> _strides) {
    assert(_shape.size() == _strides.size());
    auto dims = _shape.size();
    shape = std::vector(_shape);
    strides = std::vector(_strides);
  }

  Tensor(Tensor &&other) {
    data = std::move(other.data);
    strides = std::move(other.strides);
    shape = std::move(other.shape);
  }

  void swap(Tensor &other) {
    data.swap(other.data);
    strides.swap(other.strides);
    shape.swap(other.shape);
  }

  template <typename T>
  void resetData(T value = 0) {
    data.assign(size(), value);
  }

  size_t size() { return data.size(); }

  void clear() {
    data.clear();
    strides.clear();
    shape.clear();
  }
  void dump() {
    llvm::errs() << "shape: ";
    for (auto i : shape) {
      llvm::errs() << i << ", ";
    }
    llvm::errs() << "\n";
    llvm::errs() << "strides: ";
    for (auto i : strides) {
      llvm::errs() << i << ", ";
    }
    llvm::errs() << "\n";
    dump(llvm::errs());
  }

  void dump(raw_ostream &os, int dim = 0, size_t offset = 0,
            bool firstRow = true, bool lastRow = false) {
#define MAX_ITMES 10
    if (dim == shape.size() - 1) {
      if (firstRow)
        os << "[";
      else
        os.indent(dim + 1);
      for (int i = 0, n = shape[dim]; i < n && i < MAX_ITMES; i++) {
        os << data[offset + strides[dim] * i] << ", ";
      }
      if (shape[dim] > MAX_ITMES)
        os << "...";
      if (lastRow)
        os << "]";
      os << "\n";
      return;
    }

    if (!firstRow)
      os.indent(dim);
    os << "[";

    for (int i = 0, n = shape[dim]; i < n && i < MAX_ITMES; i++) {
      dump(os, dim + 1, offset + strides[dim] * i, i == 0,
           i == n - 1 || i == 4);
    }
    if (shape[dim] > MAX_ITMES)
      os << "...";
    os.indent(dim) << "]\n";
#undef MAX_ITMES
  }
};

template <typename DType>
class TensorManipulation {

  using Dim2Type = std::array<int64_t, 2>;
  using DimType = std::variant<int64_t, Dim2Type>;

  Tensor<DType> target;
  Tensor<DType> source;
  SmallVector<int64_t> copySize;
  SmallVector<DimType> compDims;

public:
  template <typename T>
  TensorManipulation(const std::vector<DType> &source,
                     std::initializer_list<T> shape)
      : source(source, shape){};

  template <typename T>
  TensorManipulation(const std::vector<DType> &source, ArrayRef<T> shape)
      : source(source, shape){};

  std::vector<DType> getData() { return std::move(source.getData()); }
  Tensor<DType> getTensor() { return std::move(source); }

  template <typename... Args>
  TensorManipulation &reshape(Args... args) {
    (dimRecord(args), ...);
    return reshape();
  }

  template <typename... Args>
  TensorManipulation &resize(Args... args) {
    (dimRecord(args), ...);
    return resize();
  }

  template <typename... Args>
  TensorManipulation &slice(Args... args) {
    (dimRecord(args), ...);
    return slice();
  }

  template <typename... Args>
  TensorManipulation &transpose(Args... args) {
    (dimRecord(args), ...);
    return transpose();
  }

private:
  template <typename T>
  void dimRecord(std::array<T, 2> rank) {
    compDims.push_back(std::array<int64_t, 2>{rank[0], rank[1]});
  }

  template <typename T>
  void dimRecord(T rank) {
    compDims.push_back(rank);
  }

  TensorManipulation &reset() {
    copySize.clear();
    compDims.clear();
    return *this;
  }

  TensorManipulation &resize() {
    assert(compDims.size() == source.getShape().size());

    SmallVector<int64_t> flattenShape;
    SmallVector<int64_t> shape;
    for (int i = 0, n = compDims.size(); i < n; i++) {
      if (auto *value = std::get_if<int64_t>(&compDims[i])) {
        flattenShape.push_back(*value);
        shape.push_back(*value);
      } else {
        auto pack = std::get<Dim2Type>(compDims[i]);
        flattenShape.push_back(pack[0] * pack[1]);
        shape.push_back(pack[0]);
        shape.push_back(pack[1]);
      }
      copySize.push_back(std::min(flattenShape.back(), source.getShape()[i]));
    }
    target.reshape(flattenShape);
    target.resetData(0);
    doCopy();
    target.reshape(shape);
    source.swap(target);
    return reset();
  }

  TensorManipulation &reshape() {
    SmallVector<int64_t> shape;
    size_t size = 1;
    for (auto &rank : compDims) {
      auto value = std::get<int64_t>(rank);
      shape.push_back(value);
      size *= value;
    }
    assert(size == source.size() && "the element size does not match.");
    source.reshape(shape);
    return reset();
  }

  TensorManipulation &slice() {
    assert(compDims.size() == source.getShape().size() &&
           "dimension does not match.");
    int64_t offset = 0;
    SmallVector<int64_t> shape;
    for (auto [index, rank] : llvm::enumerate(compDims)) {
      if (auto *value = std::get_if<int64_t>(&rank)) {
        shape.push_back(*value);
      } else {
        auto pack = std::get<Dim2Type>(rank);
        offset += source.getStrides()[index] * pack[0];
        shape.push_back(pack[1]);
      }
    }
    target.reshape(shape);
    target.resetData(0);
    copySize = shape;
    doCopy(0, 0, offset);
    source.swap(target);
    return reset();
  }

  TensorManipulation &transpose() {
    assert(compDims.size() == source.getShape().size() &&
           "dimension does not match.");
    SmallVector<int64_t> shape;
    SmallVector<int64_t> strides;
    for (auto &_dim : compDims) {
      auto dim = std::get<int64_t>(_dim);
      strides.push_back(source.getStrides()[dim]);
      shape.push_back(source.getShape()[dim]);
    }

    copySize.push_back(1);
    for (int64_t i = compDims.size() - 1; i >= 0; i--) {
      if (std::get<int64_t>(compDims[i]) == i && copySize.size() == 1) {
        copySize[0] *= shape[i];
        continue;
      } else {
        copySize.push_back(shape[i]);
      }
    }
    copySize = SmallVector<int64_t>(llvm::reverse(copySize));

    target.reshape(shape);
    target.resetData(0);

    source.reshape(shape);
    source.setStrides(strides);
    doCopy();
    source.swap(target);
    return reset();
  }

private:
  void doCopy(int dim = 0, size_t dest_offset = 0, size_t source_offset = 0) {
    if (dim == copySize.size() - 1) {
      std::memcpy(&target[dest_offset], &source[source_offset],
                  copySize[dim] * sizeof(DType));
      return;
    }

    for (int i = 0, n = copySize[dim]; i < n; i++) {
      doCopy(dim + 1, dest_offset + i * target.getStrides()[dim],
             source_offset + i * source.getStrides()[dim]);
    }
  }
};

} // namespace tpu_mlir

#endif
