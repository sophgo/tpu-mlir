//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

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

namespace py = pybind11;

class PyCallBack : public CallBack {
public:
  explicit PyCallBack(py::function &func) : run_(func) {}
  void run(std::string layer_name) { run_(layer_name); }
  py::function run_;
};

struct quant_brief_info {
  std::string dtype;
  std::string shape;
  float scale;
  int zp;
};

// Warning: buffer in C++. New inference will erase old output
static py::array getPyArray(std::shared_ptr<std::vector<float>> ptr,
                            const std::vector<int64_t> &shape) {
  auto shared_ptr_ptr = new std::shared_ptr<std::vector<float>>(std::move(ptr));
  py::capsule delete_shared_ptr_ptr(shared_ptr_ptr, [](void *ptr) {
    delete reinterpret_cast<std::shared_ptr<std::vector<float>> *>(ptr);
  });
  return py::array_t<float>(shape, (*shared_ptr_ptr)->data(),
                            delete_shared_ptr_ptr);
}
