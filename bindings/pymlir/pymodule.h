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

  nb::dict getAllTensor();

  void before_invoke(nb::object func);

  void after_invoke(nb::object func);

  void clear_hooks();

  static void set_mem_mode(std::string mem_mode);

  void set_tensor(
      std::string name,
      nb::ndarray<float> data,
      std::vector<int64_t> shape);

  void set_tensor_from_int(
      std::string name,
      nb::ndarray<float> data,
      std::vector<int64_t> shape);

  // Warning: using copy in python
  nb::ndarray<float, nb::numpy> get_tensor(std::string name);

  // Tip: not using copy in python, since independent mem
  nb::ndarray<float, nb::numpy> get_fp32_tensor(std::string name);

  struct quant_brief_info format_tensor_qinfo(std::string name);

  void invoke(bool fixed_to_float);
  void fake_quant_weight();

  nb::ndarray<float, nb::numpy> invoke_at(const std::string name);

  nb::ndarray<float, nb::numpy> backward_weight_at(
      const std::string name, const std::string weight_name,
      nb::ndarray<float> grd_dst);

  void invoke_from(const std::string name);

public:
  nb::list all_tensor_names;
  nb::list all_weight_names;
  nb::list input_names;
  nb::list output_names;
  static std::string gmem_mode_str_;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  std::string weightFilePath_;
  std::unique_ptr<ModuleInterpreter> interpreter_;
};
