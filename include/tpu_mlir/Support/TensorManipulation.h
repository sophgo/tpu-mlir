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
struct TensorT {
  SmallVector<int64_t> strides;
  SmallVector<int64_t> shape;
  std::vector<DType> data;

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
    data.resize(size);
    setStrides();
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

  void swap(TensorT &other) {
    data.swap(other.data);
    strides.swap(other.strides);
    shape.swap(other.shape);
  }

  DType &operator[](size_t n) { return data[n]; }

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
class Tensor {
  using Dim2Type = std::array<int64_t, 2>;
  using DimType = std::variant<int64_t, Dim2Type>;

  TensorT<DType> exchange;
  TensorT<DType> storage;
  SmallVector<int64_t> copySize;
  SmallVector<DimType> compDims;

public:
  template <typename... Args>
  Tensor &reshape(Args... args) {
    (dimRecord(args), ...);
    return reshape();
  }

  template <typename... Args>
  Tensor &resize(Args... args) {
    (dimRecord(args), ...);
    return resize();
  }

  template <typename... Args>
  Tensor &slice(Args... args) {
    (dimRecord(args), ...);
    return slice();
  }

  template <typename... Args>
  Tensor &transpose(Args... args) {
    (dimRecord(args), ...);
    return transpose();
  }

public:
  Tensor() = default;

  template <typename T>
  Tensor(const std::vector<DType> &source) {
    storage.data = source;
  }

  Tensor(const TensorT<DType> &&source) { storage = std::move(source); }

  template <typename T>
  Tensor(const std::vector<DType> &source, std::initializer_list<T> shape) {
    for (auto rank : shape) {
      storage.shape.push_back(rank);
    }
    storage.setStrides();
    storage.data = source;
  }

  template <typename T>
  Tensor(const std::vector<DType> &source, ArrayRef<T> shape) {
    for (auto rank : shape) {
      storage.shape.push_back(rank);
    }
    storage.setStrides();
    storage.data = source;
  }

  Tensor(const Tensor &&other) { storage = std::move(other.storage); }

  std::vector<DType> &getData() { return storage.data; }
  SmallVector<int64_t> &getShape() { return storage.shape; }
  SmallVector<int64_t> &getStrides() { return storage.strides; }

  DType &operator[](size_t n) { return storage.data[n]; }

  size_t size() { return storage.size(); }
  void dump() { return storage.dump(); }

  template <typename NewDType>
  Tensor<NewDType> asDType() {
    assert(sizeof(DType) * storage.shape.back() >= sizeof(NewDType));
    assert((sizeof(DType) * storage.shape.back() % sizeof(NewDType)) == 0);

    auto new_shape = storage.shape;

    float scale = (float)sizeof(DType) / sizeof(NewDType);
    new_shape.back() *= scale;
    TensorT<NewDType> outData;
    outData.reshape(new_shape);
    std::memcpy(outData.data.data(), storage.data.data(),
                size() * sizeof(DType));
    Tensor<NewDType> out(std::move(outData));
    return std::move(out);
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

  Tensor &reset() {
    copySize.clear();
    compDims.clear();
    return *this;
  }

  Tensor &resize() {
    assert(compDims.size() == storage.shape.size());
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
      copySize.push_back(std::min(flattenShape.back(), storage.shape[i]));
    }
    exchange.reshape(flattenShape);
    exchange.resetData(0);
    doCopy();
    exchange.reshape(shape);
    storage.swap(exchange);
    return reset();
  }

  Tensor &reshape() {
    SmallVector<int64_t> shape;
    size_t size = 1;
    for (auto &rank : compDims) {
      auto value = std::get<int64_t>(rank);
      size *= value;
      shape.push_back(value);
    }
    assert(size == storage.size() && "the element size does not match.");
    storage.reshape(shape);
    return reset();
  }

  Tensor &slice() {
    assert(compDims.size() == storage.shape.size() &&
           "dimension does not match.");
    int64_t offset = 0;
    SmallVector<int64_t> shape;
    for (auto [index, rank] : llvm::enumerate(compDims)) {
      if (auto *value = std::get_if<int64_t>(&rank)) {
        shape.push_back(*value);
      } else {
        auto pack = std::get<Dim2Type>(rank);
        offset += storage.strides[index] * pack[0];
        shape.push_back(pack[1]);
      }
    }
    exchange.reshape(shape);
    exchange.resetData(0);
    copySize = shape;
    doCopy(0, 0, offset);
    storage.swap(exchange);
    return reset();
  }

  Tensor &transpose() {
    assert(compDims.size() == storage.shape.size() &&
           "dimension does not match.");
    SmallVector<int64_t> shape;
    SmallVector<int64_t> strides;
    for (auto &_dim : compDims) {
      auto dim = std::get<int64_t>(_dim);
      strides.push_back(storage.strides[dim]);
      shape.push_back(storage.shape[dim]);
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

    exchange.reshape(shape);
    exchange.resetData(0);

    storage.reshape(shape);
    storage.setStrides(strides);
    doCopy();
    storage.swap(exchange);
    return reset();
  }

  void doCopy(int dim = 0, size_t dest_offset = 0, size_t source_offset = 0) {
    if (dim == copySize.size() - 1) {
      std::memcpy(&exchange[dest_offset], &storage[source_offset],
                  copySize[dim] * sizeof(DType));
      return;
    }

    for (int i = 0, n = copySize[dim]; i < n; i++) {
      doCopy(dim + 1, dest_offset + i * exchange.strides[dim],
             source_offset + i * storage.strides[dim]);
    }
  }

  template <typename T>
  Tensor &constBinary(T other, std::function<DType(DType, T)> &&func) {
    static_assert(std::is_integral<DType>::value);
    static_assert(std::is_integral<T>::value);
    for (size_t i = 0, n = size(); i < n; i++) {
      getData()[i] = func(getData()[i], other);
    }
    return *this;
  }

public:
  template <typename T>
  Tensor &operator<<=(T leftShift) {
    return constBinary<T>(leftShift, [](DType a, T b) { return a << b; });
  }

  template <typename T>
  Tensor &operator>>=(T rightShift) {
    return constBinary<T>(rightShift, [](DType a, T b) { return a >> b; });
  }

  template <typename T>
  Tensor &operator|=(T other) {
    return constBinary<T>(other, [](DType a, T b) { return a | b; });
  }

  template <typename T>
  Tensor &operator&=(T other) {
    return constBinary<T>(other, [](DType a, T b) { return a & b; });
  }

  Tensor &operator&=(Tensor &other) {
    assert(other.size() == size());
    auto otherData = other.getData();
    for (size_t i = 0, n = size(); i < n; i++) {
      storage.data[i] &= otherData[i];
    }
    return *this;
  }

  Tensor &operator|=(Tensor &other) {
    assert(other.size() == size());
    auto otherData = other.getData();
    for (size_t i = 0, n = size(); i < n; i++) {
      storage.data[i] |= otherData[i];
    }
    return *this;
  }
};

} // namespace tpu_mlir

#endif
