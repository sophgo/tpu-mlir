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
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <stdio.h>
#include <iostream>
#include "tpuv7_rt.h"
#include "tpuv7_modelrt.h"
int sg_device_init = 0;
namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

struct PythonTensor {
  PythonTensor(tpuRtDataType_t dtype_, const char *name_, float scale_,
               int zero_point_, tpuRtShape_t shape, void *data_) {
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
  void fixDtype(tpuRtDataType_t fmt) {
    switch (fmt) {
    case TPU_FLOAT32:
      pytype = py::dtype("single");
      dtype = "f32";
      break;
    case TPU_INT8:
      pytype = py::dtype("int8");
      dtype = "i8";
      break;
    case TPU_UINT8:
      pytype = py::dtype("uint8");
      dtype = "u8";
      break;
    case TPU_INT4:
      pytype = py::dtype("int8");
      dtype = "i4";
      break;
    case TPU_UINT4:
      pytype = py::dtype("uint8");
      dtype = "u4";
      break;
    case TPU_INT16:
      pytype = py::dtype("int16");
      dtype = "i16";
      break;
    case TPU_UINT16:
      pytype = py::dtype("uint16");
      dtype = "u16";
      break;
    case TPU_INT32:
      pytype = py::dtype("int32");
      dtype = "i32";
      break;
    case TPU_UINT32:
      pytype = py::dtype("uint32");
      dtype = "u32";
      break;
    case TPU_BFLOAT16:
      // numpy has no bf16 type, use uint16 instread of bf16.
      pytype = py::dtype("uint16");
      dtype = "bf16";
      break;
    case TPU_FLOAT16:
      pytype = py::dtype("float16");
      dtype = "f16";
      break;
    default:
      printf("error, tpuRtDataType_t : %d\n", fmt);
      assert(0);
    }
  }
};

struct PythonNet {
  PythonNet(tpuRtNet_t *net, const char *netname, int stage = 0) {
    m_net = *net;
    tpuRtStreamCreate(&m_stream);
    name = std::string(netname);
    m_info = tpuRtGetNetInfo(m_net, netname);
    num_input = m_info.input.num;
    num_output = m_info.output.num;
    for (int i = 0; i < num_input; i++) {
      auto &shape = m_info.stages[stage].input_shapes[i];
      auto &dtype = m_info.input.dtypes[i];
      input_shapes.push_back(shape);
      auto size = shapeCount(shape) * dataTypeSize(dtype);
      void *data;
      tpuRtMallocHost(&data, size);
      assert(data != nullptr);
      inputs.push_back(std::make_shared<PythonTensor>(
          dtype, m_info.input.names[i], m_info.input.scales[i],
          m_info.input.zero_points[i], shape, data));
      input_datas.push_back(data);
    }
    for (int i = 0; i < num_output; i++) {
      auto &shape = m_info.stages[stage].output_shapes[i];
      auto &dtype = m_info.output.dtypes[i];
      output_shapes.push_back(shape);
      auto size = shapeCount(shape) * dataTypeSize(dtype);
      void *data;
      tpuRtMallocHost(&data, size);
      assert(data != nullptr);
      outputs.push_back(std::make_shared<PythonTensor>(
          dtype, m_info.output.names[i], m_info.output.scales[i],
          m_info.output.zero_points[i], shape, data));
      output_datas.push_back(data);
    }
  }
  ~PythonNet() {
    for (auto &data : input_datas) {
      if (data != nullptr) {
        tpuRtFreeHost(data);
      }
    }
    for (auto &data : output_datas) {
      if (data != nullptr) {
        tpuRtFreeHost(data);
      }
    }
    tpuRtStreamDestroy(m_stream);
    //tpuRtFreeDevice(0);
  }

  uint64_t shapeCount(tpuRtShape_t shape) {
    uint64_t count = 1;
    for (int i = 0; i < shape.num_dims; i++) {
	count *= shape.dims[i];
    }
    return count;
  }

  int dataTypeSize(tpuRtDataType_t dtype) {
    switch (dtype) {
      case TPU_FLOAT32:
      case TPU_INT32:
      case TPU_UINT32:
        return 4;
      case TPU_FLOAT16:
      case TPU_BFLOAT16:
      case TPU_INT16:
        return 2;
      case TPU_INT8:
      case TPU_UINT8:
        return 1;
      case TPU_INT4:
      case TPU_UINT4:
        return 1;  // need modify ?  to do
      default:
        return 4;
    }
  }

  void mallocTensor(std::vector<tpuRtTensor_t> &tensor,
                                   const tpuRtIOInfo_t &info,
                                   const tpuRtShape_t *shape) {
    for (int i = 0; i < info.num; ++i) {
      tensor[i].dtype = info.dtypes[i];
      tensor[i].shape.num_dims = shape[i].num_dims;
      memcpy(tensor[i].shape.dims, shape[i].dims,
             sizeof(int) * shape[i].num_dims);
      tpuRtMalloc(&tensor[i].data, shapeCount(shape[i]) * dataTypeSize(info.dtypes[i]), 0);
    }
  }

  void freeTensor(std::vector<tpuRtTensor_t> &tensor) {
    for (auto iter : tensor) {
      tpuRtFree(&iter.data, 0);
    }
  }

  void forward() {
    /*auto ret = tpuRtLaunchNet(
        p_net, name.c_str(), input_datas.data(), input_shapes.data(),
        num_input, output_datas.data(), output_shapes.data(), num_output, true);
    assert(true == ret);*/
    std::cout << "PyRuntime.Net->forward()" << std::endl;
  }

  std::vector<std::vector<size_t>>
  forward_dynamic(std::vector<std::vector<size_t>> dyn_input_shapes) {
    auto input_shapes_ = input_shapes;
    auto output_shapes_ = output_shapes;
    auto num_inputs = input_shapes.size();
    assert(num_inputs == dyn_input_shapes.size());
    for (int i = 0; i < num_inputs; i++) {
      auto &in_ = input_shapes_[i];
      auto &dyn_in_ = dyn_input_shapes[i];
      assert(in_.num_dims == dyn_in_.size());
      for (int j = 0; j < in_.num_dims; j++) {
        in_.dims[j] = dyn_in_[j];
      }
    }
    std::cout << "Pyruntime.Net->forward_dynamic >>>>>>>" << std::endl;
    auto net_info = tpuRtGetNetInfo(m_net, name.c_str());
    std::vector<tpuRtTensor_t> input_tensors(net_info.input.num);
    std::vector<tpuRtTensor_t> output_tensors(net_info.output.num);
    mallocTensor(input_tensors, net_info.input, net_info.stages[0].input_shapes);
    mallocTensor(output_tensors, net_info.output, net_info.stages[0].output_shapes);
    for (int i = 0; i < net_info.input.num; ++i) {
        size_t data_size = shapeCount(input_shapes[i]) * dataTypeSize(input_tensors[i].dtype);
        tpuRtMemcpyS2D(input_tensors[i].data, (void *)input_datas[i], data_size);
    }
    tpuRtLaunchNet(m_net, input_tensors.data(), output_tensors.data(), name.c_str(), m_stream);

    // copy output
    for (int idx = 0; idx < net_info.output.num; idx++) {
      size_t data_size = shapeCount(output_shapes[idx]) * dataTypeSize(output_tensors[idx].dtype);
      tpuRtMemcpyD2S((void *)output_datas[idx], output_tensors[idx].data, data_size);
    }
    //free dev mem
    freeTensor(input_tensors);
    freeTensor(output_tensors);
    std::vector<std::vector<size_t>> dyn_output_shapes;
    for (auto &o : output_tensors) {
      std::vector<size_t> shape;
      for (int i = 0; i < o.shape.num_dims; i++) {
        shape.push_back(o.shape.dims[i]);
      }
      dyn_output_shapes.push_back(shape);
    }
    return dyn_output_shapes;
  }

  void printNetworkInfo(tpuRtNetInfo_t *info) {
    printf("++++++++++++++ net info ++++++++++++++\n");
    printf("net name: %s\n", info->name);
    printf("is dynamic:%d\n", info->is_dynamic);
    printf("input num:%d\n", info->input.num);
    for (int i = 0; i < info->input.num; i++) {
      printf("input:[%s], type:[%d], scale:[%f], zero_point:[%d]\n",
             info->input.names[i], info->input.dtypes[i], info->input.scales[i],
             info->input.zero_points[i]);
    }
    printf("output num:%d\n", info->output.num);
    for (int i = 0; i < info->output.num; i++) {
      printf("output:[%s], type:[%d], scale:[%f], zero_point:[%d]\n",
             info->output.names[i], info->output.dtypes[i],
             info->output.scales[i], info->output.zero_points[i]);
    }

    printf("stage num:%d\n", info->stage_num);
    for (int i = 0; i < info->stage_num; i++) {
      printf("-----------------stage[%d]-------------------\n", i);
      for (int j = 0; j < info->input.num; j++) {
        printf("input[%s], shape:[ ", info->input.names[j]);
        for (int k = 0; k < info->stages[i].input_shapes[j].num_dims; k++) {
          printf("%d ", info->stages[i].input_shapes[j].dims[k]);
        }
        printf("]\n");
      }
      for (int j = 0; j < info->output.num; j++) {
        printf("output[%s], shape:[ ", info->output.names[j]);
        for (int k = 0; k < info->stages[i].output_shapes[j].num_dims; k++) {
          printf("%d ", info->stages[i].output_shapes[j].dims[k]);
        }
        printf("]\n");
      }
    }
    printf("================ net info ===============\n");
  }

  void dump() { printNetworkInfo(&m_info); }

  std::string name;
  int num_input;
  int num_output;
  std::vector<std::shared_ptr<PythonTensor>> inputs;
  std::vector<std::shared_ptr<PythonTensor>> outputs;

private:
  PythonNet() {}
  tpuRtNet_t m_net;
  tpuRtNetInfo_t m_info;
  tpuRtStream_t m_stream;
  std::vector<tpuRtShape_t> input_shapes;
  std::vector<tpuRtShape_t> output_shapes;
  std::vector<void *> input_datas;
  std::vector<void *> output_datas;
};

struct PythonModel {
  PythonModel(const std::string &model_file, int dev_id,
	      const std::string &decrypt_lib) {
    tpuRtStatus_t ret = tpuRtSuccess;
    m_dev_id = dev_id;
    if (!sg_device_init) {
      tpuRtInit();
      tpuRtSetDevice(dev_id);
      sg_device_init = 1;
    }
    tpuRtCreateNetContext(&m_context);
    ret = tpuRtLoadNet(model_file.c_str(), m_context, &m_net);
    char **net_names = NULL;
    m_net_num = tpuRtGetNetNames(m_net, &net_names);
    for (int i = 0; i < m_net_num; i++) {
      networks.push_back(net_names[i]);
    }
    tpuRtFreeNetNames(net_names);
  }

    std::shared_ptr<PythonNet> Net(const char *net) {
      return std::make_shared<PythonNet>(&m_net, net);
  }

  ~PythonModel() {
    tpuRtUnloadNet(m_net);
    tpuRtDestroyNetContext(m_context);
  }

  std::vector<const char *> networks;

private:
  PythonModel() {}
  uint32_t chip_id;
  tpuRtNet_t m_net;
  tpuRtNetContext_t m_context;
  int m_net_num;
  int m_dev_id = 0;
};

PYBIND11_MODULE(pyruntime_tpuv7, m) {
  py::class_<PythonTensor, std::shared_ptr<PythonTensor>>(m, "Tensor")
      .def_readonly("name", &PythonTensor::name)
      .def_readonly("qscale", &PythonTensor::qscale)
      .def_readonly("qzero_point", &PythonTensor::qzero_point)
      .def_readonly("dtype", &PythonTensor::dtype)
      .def_readwrite("data", &PythonTensor::data);

  py::class_<PythonNet, std::shared_ptr<PythonNet>>(m, "Net")
      .def("forward", &PythonNet::forward)
      .def("forward_dynamic", &PythonNet::forward_dynamic)
      .def("dump", &PythonNet::dump)
      .def_readonly("name", &PythonNet::name)
      .def_readonly("num_input", &PythonNet::num_input)
      .def_readonly("num_output", &PythonNet::num_output)
      .def_readwrite("inputs", &PythonNet::inputs)
      .def_readwrite("outputs", &PythonNet::outputs);

  py::class_<PythonModel>(m, "Model")
      .def(py::init<const std::string &, int, const std::string &>(),
           py::arg("model_file"), py::arg("device_id") = 0,
           py::arg("decrypt_lib") = "")
      .def("Net", &PythonModel::Net)
      .def_readonly("networks", &PythonModel::networks);

  py::scoped_ostream_redirect output{std::cerr,
                                     py::module::import("sys").attr("stderr")};
}
