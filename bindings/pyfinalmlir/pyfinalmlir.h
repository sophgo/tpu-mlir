//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <memory>
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/BM168xEvaluator.h"

namespace nb = nanobind;

class py_final_module {
public:
  py_final_module() {}
  ~py_final_module();
  void load(std::string filename);
  nb::dict getAllTensor();
  void set_tensor(
      std::string name,
      nb::ndarray<float> data);
  void set_tensor_from_int(
      std::string name,
      nb::ndarray<float> data);
  void invoke();

public:
  nb::list all_tensor_names;
  nb::list all_weight_names;
  nb::list input_names;
  nb::list output_names;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  std::unique_ptr<tpu::BM168xEvaluator> evaluator_;
};

static nb::ndarray<float, nb::numpy> getPyArray(std::shared_ptr<std::vector<float>> ptr,
                            const std::vector<int64_t> &shape) {
  auto shared_ptr_ptr = new std::shared_ptr<std::vector<float>>(std::move(ptr));
  auto delete_shared_ptr_ptr = nb::capsule(shared_ptr_ptr, [](void *ptr) noexcept {
    delete reinterpret_cast<std::shared_ptr<std::vector<float>> *>(ptr);
  });

  std::vector<size_t> shape_vec(shape.begin(), shape.end());

  return nb::ndarray<float, nb::numpy>((*shared_ptr_ptr)->data(), shape_vec.size(), shape_vec.data(), delete_shared_ptr_ptr);
}
