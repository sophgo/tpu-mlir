//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "pymlir.h"

class py_module {
public:
  py_module() {}
  ~py_module();
  void load(std::string filename);

  py::dict getAllTensor();

  void before_invoke(py::function func);

  void after_invoke(py::function func);

  void clear_hooks();

  static void set_mem_mode(std::string mem_mode);

  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data,
      std::vector<int64_t> shape);

  void set_tensor_from_int(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data,
      std::vector<int64_t> shape);

  // Warning: using copy in python
  py::array get_tensor(std::string name);

  // Tip: not using copy in python, since independent mem
  py::array get_fp32_tensor(std::string name);

  struct quant_brief_info format_tensor_qinfo(std::string name);

  void invoke(bool fixed_to_float);
  void fake_quant_weight();

  py::array invoke_at(const std::string name);

  py::array backward_weight_at(
      const std::string name, const std::string weight_name,
      py::array_t<float, py::array::c_style | py::array::forcecast> grd_dst);

  void invoke_from(const std::string name);

public:
  py::list all_tensor_names;
  py::list all_weight_names;
  py::list input_names;
  py::list output_names;
  static std::string gmem_mode_str_;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  std::string weightFilePath_;
  std::unique_ptr<ModuleInterpreter> interpreter_;
};
