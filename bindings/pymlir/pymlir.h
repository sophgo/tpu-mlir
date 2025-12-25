//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <vector>
#include <memory>

// -------------
// pure C++ code
// -------------
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "tpu_mlir/InitAll.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "host/ModuleInterpreter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace tpu_mlir;

typedef std::map<std::string, std::shared_ptr<std::vector<float>>> tensor_map_t;
typedef std::map<std::string, std::vector<int64_t>> shape_map_t;

// ----------------
// Python interface
// ----------------

namespace nb = nanobind;

class PyCallBack : public CallBack {
public:
  explicit PyCallBack(nb::object &func) : run_(func) {}
  void run(std::string layer_name) { run_(layer_name); }
  nb::object run_;
};

struct quant_brief_info {
  std::string dtype;
  std::string shape;
  float scale;
  int zp;
};

// Warning: buffer in C++. New inference will erase old output
static nb::ndarray<> getPyArray(std::shared_ptr<std::vector<float>> ptr,
                              const std::vector<int64_t> &shape) {
  auto shared_ptr_ptr = new std::shared_ptr<std::vector<float>>(std::move(ptr));
  nb::capsule delete_shared_ptr_ptr(shared_ptr_ptr, [](void *ptr) noexcept {
    delete reinterpret_cast<std::shared_ptr<std::vector<float>> *>(ptr);
  });
  std::vector<size_t> shape_sz(shape.begin(), shape.end());
  // Create contiguous strides in elements (float)
  std::vector<size_t> strides(shape_sz.size(), sizeof(float));
  for (int i = static_cast<int>(shape_sz.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape_sz[i + 1];
  }
  std::vector<int64_t> strides_int64(strides.begin(), strides.end());
  return nb::ndarray((*shared_ptr_ptr)->data(), shape_sz.size(), shape_sz.data(), delete_shared_ptr_ptr, strides_int64.data(), nb::dtype<float>());
}
