//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "bmruntime_interface.h"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

struct PythonTensor {
  PythonTensor(bm_data_type_t dtype_, const char *name_, float scale_,
               int zero_point_, bm_shape_t shape, void *data_) {
    name = std::string(name_);
    qscale = scale_;
    qzero_point = zero_point_;
    std::vector<size_t> s(shape.dims, shape.dims + shape.num_dims);
    fixDtype(dtype_);
    data = py::array(pytype, s, data_, py::cast(*this));
  }

  std::string name;
  std::string dtype; // f32/f16/bf16/i8/i16/i32/u8/u16/u32
  float qscale;
  int qzero_point;
  py::array data;

private:
  py::dtype pytype;
  void fixDtype(bm_data_type_t fmt) {
    switch (fmt) {
    case BM_FLOAT32:
      pytype = py::dtype("single");
      dtype = "f32";
      break;
    case BM_INT8:
      pytype = py::dtype("int8");
      dtype = "i8";
      break;
    case BM_UINT8:
      pytype = py::dtype("uint8");
      dtype = "u8";
      break;
    case BM_INT16:
      pytype = py::dtype("int16");
      dtype = "i16";
      break;
    case BM_UINT16:
      pytype = py::dtype("uint16");
      dtype = "u16";
      break;
    case BM_INT32:
      pytype = py::dtype("int32");
      dtype = "i32";
      break;
    case BM_UINT32:
      pytype = py::dtype("uint32");
      dtype = "u32";
      break;
    case BM_BFLOAT16:
      // numpy has no bf16 type, use uint16 instread of bf16.
      pytype = py::dtype("uint16");
      dtype = "bf16";
      break;
    case BM_FLOAT16:
      pytype = py::dtype("float16");
      dtype = "f16";
      break;
    default:
      assert(0);
    }
  }
};

struct PythonNet {
  PythonNet(void *bmrt, const char *netname, bm_handle_t handle,
            int stage = 0) {
    p_bmrt = bmrt;
    bm_handle = handle;
    name = std::string(netname);
    info = bmrt_get_network_info(p_bmrt, netname);
    num_input = info->input_num;
    num_output = info->output_num;
    for (int i = 0; i < num_input; i++) {
      auto &shape = info->stages[stage].input_shapes[i];
      auto &dtype = info->input_dtypes[i];
      input_shapes.push_back(shape);
      auto size = bmrt_shape_count(&shape) * bmrt_data_type_size(dtype);
      void *data = malloc(size);
      assert(data != nullptr);
      inputs.push_back(std::make_shared<PythonTensor>(
          dtype, info->input_names[i], info->input_scales[i],
          info->input_zero_point[i], shape, data));
      input_datas.push_back(data);
    }
    for (int i = 0; i < num_output; i++) {
      auto &shape = info->stages[stage].output_shapes[i];
      auto &dtype = info->output_dtypes[i];
      output_shapes.push_back(shape);
      auto size = bmrt_shape_count(&shape) * bmrt_data_type_size(dtype);
      void *data = malloc(size);
      assert(data != nullptr);
      outputs.push_back(std::make_shared<PythonTensor>(
          dtype, info->output_names[i], info->output_scales[i],
          info->output_zero_point[i], shape, data));
      output_datas.push_back(data);
    }
  }
  ~PythonNet() {
    for (auto &data : input_datas) {
      if (data != nullptr) {
        free(data);
      }
    }
    for (auto &data : output_datas) {
      if (data != nullptr) {
        free(data);
      }
    }
  }

  void forward() {
    auto ret = bmrt_launch_data(
        p_bmrt, name.c_str(), input_datas.data(), input_shapes.data(),
        num_input, output_datas.data(), output_shapes.data(), num_output, true);
    assert(true == ret);
  }

  void dump() { bmrt_print_network_info(info); }

  std::string name;
  int num_input;
  int num_output;
  std::vector<std::shared_ptr<PythonTensor>> inputs;
  std::vector<std::shared_ptr<PythonTensor>> outputs;

private:
  PythonNet() {}
  void *p_bmrt;
  const bm_net_info_t *info;
  bm_handle_t bm_handle;
  std::vector<bm_shape_t> input_shapes;
  std::vector<bm_shape_t> output_shapes;
  std::vector<void *> input_datas;
  std::vector<void *> output_datas;
};

struct PythonModel {
  PythonModel(const std::string &model_file, int dev_id) {
    auto ret = bm_dev_request(&bm_handle, dev_id);
    assert(ret == 0);
    ret = bm_get_chipid(bm_handle, &chip_id);
    assert(ret == 0);
    p_bmrt = bmrt_create(bm_handle);
    assert(p_bmrt != nullptr);
    bool flag = bmrt_load_bmodel(p_bmrt, model_file.c_str());
    assert(flag == true);
    const char **net_names = NULL;
    bmrt_get_network_names(p_bmrt, &net_names);
    net_num = bmrt_get_network_number(p_bmrt);
    for (int i = 0; i < net_num; i++) {
      networks.push_back(net_names[i]);
    }
    delete net_names;
  }

  std::shared_ptr<PythonNet> Net(const char *net) {
    return std::make_shared<PythonNet>(p_bmrt, net, bm_handle);
  }

  ~PythonModel() {
    bmrt_destroy(p_bmrt);
    bm_dev_free(bm_handle);
  }

  std::vector<const char *> networks;

private:
  PythonModel() {}
  bm_handle_t bm_handle;
  uint32_t chip_id;
  void *p_bmrt;
  int net_num;
};

PYBIND11_MODULE(pyruntime_bm, m) {
  py::class_<PythonTensor, std::shared_ptr<PythonTensor>>(m, "Tensor")
      .def_readonly("name", &PythonTensor::name)
      .def_readonly("qscale", &PythonTensor::qscale)
      .def_readonly("qzero_point", &PythonTensor::qzero_point)
      .def_readonly("dtype", &PythonTensor::dtype)
      .def_readwrite("data", &PythonTensor::data);

  py::class_<PythonNet, std::shared_ptr<PythonNet>>(m, "Net")
      .def("forward", &PythonNet::forward)
      .def("dump", &PythonNet::dump)
      .def_readonly("name", &PythonNet::name)
      .def_readonly("num_input", &PythonNet::num_input)
      .def_readonly("num_output", &PythonNet::num_output)
      .def_readwrite("inputs", &PythonNet::inputs)
      .def_readwrite("outputs", &PythonNet::outputs);

  py::class_<PythonModel>(m, "Model")
      .def(py::init<const std::string &, int>(), py::arg("model_file"),
           py::arg("device_id") = 0)
      .def("Net", &PythonModel::Net)
      .def_readonly("networks", &PythonModel::networks);
}
