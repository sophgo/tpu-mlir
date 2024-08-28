//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/BM168xEvaluator.h"

namespace py = pybind11;

class py_final_module {
public:
  py_final_module() {}
  ~py_final_module();
  void load(std::string filename);
  py::dict getAllTensor();
  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data);
  void set_tensor_from_int(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data);
  void invoke();

public:
  py::list all_tensor_names;
  py::list all_weight_names;
  py::list input_names;
  py::list output_names;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  std::unique_ptr<tpu::BM168xEvaluator> evaluator_;
};

static py::array getPyArray(std::shared_ptr<std::vector<float>> ptr,
                            const std::vector<int64_t> &shape) {
  auto shared_ptr_ptr = new std::shared_ptr<std::vector<float>>(std::move(ptr));
  py::capsule delete_shared_ptr_ptr(shared_ptr_ptr, [](void *ptr) {
    delete reinterpret_cast<std::shared_ptr<std::vector<float>> *>(ptr);
  });
  return py::array_t<float>(shape, (*shared_ptr_ptr)->data(),
                            delete_shared_ptr_ptr);
}
