//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//


#include "cviruntime.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <memory>
#include <variant>

namespace nb = nanobind;

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
    createNdarray(tensor->fmt, (void *)CVI_NN_TensorPtr(tensor), shape);
  }

  std::string name;
  std::string dtype; // f32/f16/bf16/i8/i16/i32/u8/u16/u32
  float qscale;
  int zpoint;
  bool aligned = false;
  size_t size;
  size_t dsize;


  std::variant<
  nb::ndarray<nb::numpy, float>,
  nb::ndarray<nb::numpy, int8_t>,
  nb::ndarray<nb::numpy, uint8_t>,
  nb::ndarray<nb::numpy, int16_t>,
  nb::ndarray<nb::numpy, uint16_t>,
  nb::ndarray<nb::numpy, int32_t>,
  nb::ndarray<nb::numpy, uint32_t>
> data;

private:
  // Note: nanobind doesn't have py::dtype, we'll handle dtype differently
  void fixDtype(CVI_FMT fmt) {
    switch (fmt) {
    case CVI_FMT_FP32:
      dtype = "f32";
      dsize = 4;
      break;
    case CVI_FMT_INT8:
      dtype = "i8";
      dsize = 1;
      break;
    case CVI_FMT_UINT8:
      dtype = "u8";
      dsize = 1;
      break;
    case CVI_FMT_INT16:
      dtype = "i16";
      dsize = 2;
      break;
    case CVI_FMT_UINT16:
      dtype = "u16";
      dsize = 2;
      break;
    case CVI_FMT_INT32:
      dtype = "i32";
      dsize = 4;
      break;
    case CVI_FMT_UINT32:
      dtype = "u32";
      dsize = 4;
      break;
    case CVI_FMT_BF16:
      dtype = "bf16";
      dsize = 2;
      break;
    default:
      assert(0);
    }
  }


  void createNdarray(CVI_FMT dtype_, void *data_, const std::vector<size_t> &shape) {
    // 根据数据类型创建对应的ndarray
    switch (dtype_) {
    case CVI_FMT_FP32:
      data = nb::ndarray<nb::numpy, float>(static_cast<float*>(data_), shape.size(), shape.data());
      break;
    case CVI_FMT_INT8:
      data = nb::ndarray<nb::numpy, int8_t>(static_cast<int8_t*>(data_), shape.size(), shape.data());
      break;
    case CVI_FMT_UINT8:
      data = nb::ndarray<nb::numpy, uint8_t>(static_cast<uint8_t*>(data_), shape.size(), shape.data());
      break;
    case CVI_FMT_INT16:
      data = nb::ndarray<nb::numpy, int16_t>(static_cast<int16_t*>(data_), shape.size(), shape.data());
      break;
    case CVI_FMT_UINT16:
      data = nb::ndarray<nb::numpy, uint16_t>(static_cast<uint16_t*>(data_), shape.size(), shape.data());
      break;
    case CVI_FMT_INT32:
      data = nb::ndarray<nb::numpy, int32_t>(static_cast<int32_t*>(data_), shape.size(), shape.data());
      break;
    case CVI_FMT_UINT32:
      data = nb::ndarray<nb::numpy, uint32_t>(static_cast<uint32_t*>(data_), shape.size(), shape.data());
      break;
    case CVI_FMT_BF16:
      data = nb::ndarray<nb::numpy, uint16_t>(static_cast<uint16_t*>(data_), shape.size(), shape.data());
      break;
    default:
      printf("error, unsupported CVI_FMT : %d\n", dtype_);
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

  nb::object clone() {
    auto new_cvimodel = new PythonCviModel();
    int ret = CVI_NN_CloneModel(model, &new_cvimodel->model);
    if (ret != 0) {
      assert(0);
    }
    return nb::cast(new_cvimodel);
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

NB_MODULE(pyruntime_cvi, m) {
  nb::class_<PythonTensor>(m, "Tensor")
      .def_ro("name", &PythonTensor::name)
      .def_ro("qscale", &PythonTensor::qscale)
      .def_ro("qzero_point", &PythonTensor::zpoint)
      .def_ro("dtype", &PythonTensor::dtype)
      .def_ro("aligned", &PythonTensor::aligned)
      .def_ro("size", &PythonTensor::size)
      .def_rw("data", &PythonTensor::data);

  nb::class_<PythonCviModel>(m, "Model")
      .def(nb::init<const std::string &, int, bool>(), nb::arg("cvimodel"),
           nb::arg("program_id") = 0, nb::arg("output_all_tensors") = true)
      .def("forward", &PythonCviModel::forward)
      .def_rw("inputs", &PythonCviModel::inputs)
      .def_rw("outputs", &PythonCviModel::outputs);
  // Note: nanobind doesn't have scoped_ostream_redirect, we'll handle this differently
}
