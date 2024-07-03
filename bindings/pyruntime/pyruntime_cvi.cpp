//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "cviruntime.h"
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

struct PythonTensor {
  PythonTensor(CVI_TENSOR *tensor) {
    name = std::string(tensor->name);
    qscale = tensor->qscale;
    zpoint = tensor->zero_point;
    std::vector<size_t> shape;
    for (int i = 0; i < (int)tensor->shape.dim_size; i++) {
      shape.push_back(tensor->shape.dim[i]);
    }
    fixDtype(tensor->fmt);
    aligned = tensor->aligned;
    size = tensor->mem_size / dsize;
    if (aligned) {
      // for model_runner reference
      shape = {1, 1, 1, size};
    }
    data = py::array(pytype, shape, (void *)CVI_NN_TensorPtr(tensor),
                     py::cast(*this));
  }

  std::string name;
  std::string dtype; // f32/f16/bf16/i8/i16/i32/u8/u16/u32
  float qscale;
  int zpoint;
  bool aligned = false;
  size_t size;
  size_t dsize;
  py::array data;

private:
  py::dtype pytype;
  void fixDtype(CVI_FMT fmt) {
    switch (fmt) {
    case CVI_FMT_FP32:
      pytype = py::dtype("single");
      dtype = "f32";
      dsize = 4;
      break;
    case CVI_FMT_INT8:
      pytype = py::dtype("int8");
      dtype = "i8";
      dsize = 1;
      break;
    case CVI_FMT_UINT8:
      pytype = py::dtype("uint8");
      dtype = "u8";
      dsize = 1;
      break;
    case CVI_FMT_INT16:
      pytype = py::dtype("int16");
      dtype = "i16";
      dsize = 2;
      break;
    case CVI_FMT_UINT16:
      pytype = py::dtype("uint16");
      dtype = "u16";
      dsize = 2;
      break;
    case CVI_FMT_INT32:
      pytype = py::dtype("int32");
      dtype = "i32";
      dsize = 4;
      break;
    case CVI_FMT_UINT32:
      pytype = py::dtype("uint32");
      dtype = "u32";
      dsize = 4;
      break;
    case CVI_FMT_BF16:
      // numpy has no bf16 type, use uint16 instread of bf16.
      pytype = py::dtype("uint16");
      dtype = "bf16";
      dsize = 2;
      break;
    default:
      assert(0);
    }
  }
};

struct PythonCviModel {
  PythonCviModel(const std::string &model_file, int program_id,
                 bool output_all_tensors) {
    int ret = CVI_NN_RegisterModel(model_file.c_str(), &model);
    if (ret != 0) {
      assert(0);
    }
    this->config(program_id, output_all_tensors);
  }

  ~PythonCviModel() { CVI_NN_CleanupModel(model); }

  py::object clone() {
    auto new_cvimodel = new PythonCviModel();
    int ret = CVI_NN_CloneModel(model, &new_cvimodel->model);
    if (ret != 0) {
      assert(0);
    }
    return py::cast(new_cvimodel);
  }

  void config(int program_id, bool output_all_tensors) {
    CVI_NN_SetConfig(model, OPTION_PROGRAM_INDEX, program_id);
    CVI_NN_SetConfig(model, OPTION_OUTPUT_ALL_TENSORS, output_all_tensors);
    int32_t ret = CVI_NN_GetInputOutputTensors(
        model, &input_tensors, &input_num, &output_tensors, &output_num);
    if (ret != 0) {
      assert(0);
    }
    for (int i = 0; i < input_num; i++) {
      inputs.push_back(std::make_shared<PythonTensor>(&input_tensors[i]));
    }

    for (int i = 0; i < output_num; i++) {
      if (!output_tensors[i].name) {
        continue;
      }
      outputs.push_back(std::make_shared<PythonTensor>(&output_tensors[i]));
    }
  }

  void forward() {
    int ret = CVI_NN_Forward(model, input_tensors, input_num, output_tensors,
                             output_num);
    if (ret != 0) {
      assert(0);
    }
  }

  std::vector<std::shared_ptr<PythonTensor>> inputs;
  std::vector<std::shared_ptr<PythonTensor>> outputs;

private:
  PythonCviModel() {}
  CVI_MODEL_HANDLE model = nullptr;
  int32_t input_num = 0;
  int32_t output_num = 0;
  CVI_TENSOR *input_tensors = nullptr;
  CVI_TENSOR *output_tensors = nullptr;
};

PYBIND11_MODULE(pyruntime_cvi, m) {
  py::class_<PythonTensor, std::shared_ptr<PythonTensor>>(m, "Tensor")
      .def_readonly("name", &PythonTensor::name)
      .def_readonly("qscale", &PythonTensor::qscale)
      .def_readonly("qzero_point", &PythonTensor::zpoint)
      .def_readonly("dtype", &PythonTensor::dtype)
      .def_readonly("aligned", &PythonTensor::aligned)
      .def_readonly("size", &PythonTensor::size)
      .def_readwrite("data", &PythonTensor::data);

  py::class_<PythonCviModel>(m, "Model")
      .def(py::init<const std::string &, int, bool>(), py::arg("cvimodel"),
           py::arg("program_id") = 0, py::arg("output_all_tensors") = true)
      .def("forward", &PythonCviModel::forward)
      .def_readwrite("inputs", &PythonCviModel::inputs)
      .def_readwrite("outputs", &PythonCviModel::outputs);
  py::scoped_ostream_redirect output{std::cerr,
                                     py::module::import("sys").attr("stderr")};
}
