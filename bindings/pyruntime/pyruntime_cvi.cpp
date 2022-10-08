#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cviruntime.h"

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
    data = py::array(getDtype(tensor->fmt), shape, (void *)CVI_NN_TensorPtr(tensor),
                     py::cast(*this));
  }

  std::string name;
  float qscale;
  int zpoint;
  py::array data;

private:
  py::dtype getDtype(CVI_FMT fmt) {
    switch (fmt) {
      case CVI_FMT_FP32:
        return py::dtype("single");
      case CVI_FMT_INT8:
        return py::dtype("int8");
      case CVI_FMT_UINT8:
        return py::dtype("uint8");
      case CVI_FMT_INT16:
        return py::dtype("int16");
      case CVI_FMT_UINT16:
        return py::dtype("uint16");
      case CVI_FMT_INT32:
        return py::dtype("int32");
      case CVI_FMT_UINT32:
        return py::dtype("uint32");
      case CVI_FMT_BF16:
        // numpy has no bf16 type, use uint16 instread of bf16.
        return py::dtype("uint16");
      default:
        assert(0);
    }
  }
};

struct PythonCviModel {
  PythonCviModel(const std::string &model_file, int program_id, bool output_all_tensors) {
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
    int32_t ret = CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num,
                                         &output_tensors, &output_num);
    if (ret != 0) {
      assert(0);
    }
    for (int i = 0; i < input_num; i++) {
      inputs.push_back(std::make_shared<PythonTensor>(&input_tensors[i]));
    }
    for (int i = 0; i < output_num; i++) {
      outputs.push_back(std::make_shared<PythonTensor>(&output_tensors[i]));
    }
  }

  void forward() {
    int ret = CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
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
      .def_readonly("zpoint", &PythonTensor::zpoint)
      .def_readwrite("data", &PythonTensor::data);

  py::class_<PythonCviModel>(m, "Model")
      .def(py::init<const std::string &, int, bool>(),
           py::arg("cvimodel"), py::arg("program_id") = 0,
           py::arg("output_all_tensors") = true)
      .def("forward", &PythonCviModel::forward)
      .def_readwrite("inputs", &PythonCviModel::inputs)
      .def_readwrite("outputs", &PythonCviModel::outputs);
}
