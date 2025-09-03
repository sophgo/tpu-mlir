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
#include <dlfcn.h>
#include <variant>
#include "bmcpu.h"
#include "bmruntime_interface.h"

namespace nb = nanobind;

struct PythonTensor {
  PythonTensor(bm_data_type_t dtype_, const char *name_, float scale_,
               int zero_point_, bm_shape_t shape, void *data_) {
    name = std::string(name_);
    qscale = scale_;
    qzero_point = zero_point_;
    std::vector<size_t> s(shape.dims, shape.dims + shape.num_dims);
    fixDtype(dtype_);
    createNdarray(dtype_, data_, s);
  }

  std::string name;
  std::string dtype; // f32/f16/bf16/i8/i16/i32/u8/u16/u32
  float qscale;
  int qzero_point;

  // 使用 variant 来存储不同类型的 ndarray，保持类型信息
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
  void fixDtype(bm_data_type_t fmt) {
    switch (fmt) {
    case BM_FLOAT32:
      dtype = "f32";
      break;
    case BM_INT8:
      dtype = "i8";
      break;
    case BM_UINT8:
      dtype = "u8";
      break;
    case BM_INT4:
      dtype = "i4";
      break;
    case BM_UINT4:
      dtype = "u4";
      break;
    case BM_INT16:
      dtype = "i16";
      break;
    case BM_UINT16:
      dtype = "u16";
      break;
    case BM_INT32:
      dtype = "i32";
      break;
    case BM_UINT32:
      dtype = "u32";
      break;
    case BM_BFLOAT16:
      dtype = "bf16";
      break;
    case BM_FLOAT16:
      dtype = "f16";
      break;
    default:
      printf("error, bm_data_type_t : %d\n", fmt);
      assert(0);
    }
  }

  void createNdarray(bm_data_type_t dtype_, void *data_, const std::vector<size_t> &shape) {
    // 根据数据类型创建对应的ndarray
    switch (dtype_) {
    case BM_FLOAT32:
      data = nb::ndarray<nb::numpy, float>(static_cast<float*>(data_), shape.size(), shape.data());
      break;
    case BM_INT8:
      data = nb::ndarray<nb::numpy, int8_t>(static_cast<int8_t*>(data_), shape.size(), shape.data());
      break;
    case BM_UINT8:
      data = nb::ndarray<nb::numpy, uint8_t>(static_cast<uint8_t*>(data_), shape.size(), shape.data());
      break;
    case BM_INT16:
      data = nb::ndarray<nb::numpy, int16_t>(static_cast<int16_t*>(data_), shape.size(), shape.data());
      break;
    case BM_UINT16:
      data = nb::ndarray<nb::numpy, uint16_t>(static_cast<uint16_t*>(data_), shape.size(), shape.data());
      break;
    case BM_INT32:
      data = nb::ndarray<nb::numpy, int32_t>(static_cast<int32_t*>(data_), shape.size(), shape.data());
      break;
    case BM_UINT32:
      data = nb::ndarray<nb::numpy, uint32_t>(static_cast<uint32_t*>(data_), shape.size(), shape.data());
      break;
    case BM_FLOAT16:
      data = nb::ndarray<nb::numpy, uint16_t>(static_cast<uint16_t*>(data_), shape.size(), shape.data());
      break;
    case BM_BFLOAT16:
      data = nb::ndarray<nb::numpy, uint16_t>(static_cast<uint16_t*>(data_), shape.size(), shape.data());
      break;
    case BM_INT4:
    case BM_UINT4:
      // 对于4位数据类型，使用uint8_t存储，但需要特殊处理
      data = nb::ndarray<nb::numpy, uint8_t>(static_cast<uint8_t*>(data_), shape.size(), shape.data());
      break;
    default:
      printf("error, unsupported bm_data_type_t : %d\n", dtype_);
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
    auto net_info = bmrt_get_network_info(p_bmrt, name.c_str());
    std::vector<bm_tensor_t> input_tensors(net_info->input_num);
    std::vector<bm_tensor_t> output_tensors(net_info->output_num);
    bm_device_mem_t all_input_device_mem;
    bm_device_mem_t all_output_device_mem;
    if (net_info->addr_mode == ADDR_MODE_IO_TAG_FUSE) {
      /* prepare input */
      // caculate all_input_size
      size_t all_input_size = 0;
      for (int input_idx = 0; input_idx < net_info->input_num; input_idx++) {
        all_input_size +=
            bmrt_shape_count(&input_shapes_[input_idx]) *
            bmrt_data_type_size(net_info->input_dtypes[input_idx]);
      }
      // malloc all_input_device_mem
      bm_malloc_device_byte(bm_handle, &all_input_device_mem, all_input_size);
      uint64_t all_input_addr = bm_mem_get_device_addr(all_input_device_mem);
      // setup input tensors
      size_t in_offset = 0;
      for (int input_idx = 0; input_idx < net_info->input_num; input_idx++) {
        auto &input_tensor = input_tensors[input_idx];
        size_t tensor_size =
            bmrt_shape_count(&input_shapes_[input_idx]) *
            bmrt_data_type_size(net_info->input_dtypes[input_idx]);
        auto tensor_mem =
            bm_mem_from_device(all_input_addr + in_offset, tensor_size);
        bmrt_tensor_with_device(&input_tensor, tensor_mem,
                                net_info->input_dtypes[input_idx],
                                input_shapes_[input_idx]);
        bm_memcpy_s2d(bm_handle, input_tensor.device_mem,
                      (void *)input_datas[input_idx]);
        in_offset += tensor_size;
      }
      /* prepare output */
      // caculate all_output_size
      size_t all_output_size = 0;
      for (int output_idx = 0; output_idx < net_info->output_num;
           output_idx++) {
        all_output_size +=
            bmrt_shape_count(&output_shapes_[output_idx]) *
            bmrt_data_type_size(net_info->output_dtypes[output_idx]);
      }
      // malloc all_output_device_mem
      bm_malloc_device_byte(bm_handle, &all_output_device_mem, all_output_size);
      uint64_t all_output_addr = bm_mem_get_device_addr(all_output_device_mem);
      // setup output tensors
      size_t out_offset = 0;
      for (int output_idx = 0; output_idx < net_info->output_num;
           output_idx++) {
        auto &output_tensor = output_tensors[output_idx];
        size_t tensor_size =
            bmrt_shape_count(&output_shapes_[output_idx]) *
            bmrt_data_type_size(net_info->output_dtypes[output_idx]);
        auto tensor_mem =
            bm_mem_from_device(all_output_addr + out_offset, tensor_size);
        bmrt_tensor_with_device(&output_tensor, tensor_mem,
                                net_info->output_dtypes[output_idx],
                                output_shapes_[output_idx]);
        out_offset += tensor_size;
      }
    } else {
      for (int idx = 0; idx < net_info->input_num; idx++) {
        auto &tensor = input_tensors[idx];
        bmrt_tensor(&tensor, p_bmrt, net_info->input_dtypes[idx],
                    input_shapes_[idx]);
        bm_memcpy_s2d(bm_handle, tensor.device_mem, (void *)input_datas[idx]);
      }
      for (int idx = 0; idx < net_info->output_num; idx++) {
        auto &tensor = output_tensors[idx];
        bmrt_tensor(&tensor, p_bmrt, net_info->output_dtypes[idx],
                    output_shapes_[idx]);
      }
    }
    auto ret = bmrt_launch_tensor_ex(p_bmrt, name.c_str(), input_tensors.data(),
                                     net_info->input_num, output_tensors.data(),
                                     net_info->output_num, true, false);
    assert(true == ret);
    auto status = bm_thread_sync(bm_handle);
    assert(BM_SUCCESS == status);
    // free mem
    if (net_info->addr_mode == 3) {
      for (int idx = 0; idx < net_info->output_num; idx++) {
        auto &tensor = output_tensors[idx];
        bm_memcpy_d2s(bm_handle, (void *)output_datas[idx], tensor.device_mem);
      }
      bm_free_device(bm_handle, all_input_device_mem);
      bm_free_device(bm_handle, all_output_device_mem);
    } else {
      for (int idx = 0; idx < net_info->output_num; idx++) {
        auto &tensor = output_tensors[idx];
        bm_memcpy_d2s(bm_handle, (void *)output_datas[idx], tensor.device_mem);
        bm_free_device(bm_handle, tensor.device_mem);
      }
      for (int idx = 0; idx < num_inputs; idx++) {
        auto &tensor = input_tensors[idx];
        bm_free_device(bm_handle, tensor.device_mem);
      }
    }
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
  PythonModel(const std::string &model_file, int dev_id,
              const std::string &decrypt_lib) {
    auto ret = bm_dev_request(&bm_handle, dev_id);
    assert(ret == 0);
    ret = bm_get_chipid(bm_handle, &chip_id);
    assert(ret == 0);
    p_bmrt = bmrt_create(bm_handle);
    assert(p_bmrt != nullptr);
    bool flag = true;
    if (decrypt_lib.empty()) {
      flag = bmrt_load_bmodel(p_bmrt, model_file.c_str());
    } else {
      // flag = bmrt_load_bmodel_with_decrypt_lib(p_bmrt, model_file.c_str(),
      // decrypt_lib.c_str());
      auto handle = dlopen(decrypt_lib.c_str(), RTLD_LAZY);
      assert(handle != nullptr);
      auto f = (decrypt_func)dlsym(handle, "decrypt");
      assert(f != nullptr);
      flag = bmrt_load_bmodel_with_decrypt(p_bmrt, model_file.c_str(), f);
      dlclose(handle);
    }
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

struct PyCpuLayer {
  PyCpuLayer() { bmcpu_handle = bmcpu_init(); }

  std::vector<std::vector<int>> forward(
      int op_type, nb::bytes param_py, int param_size,
      std::vector<std::vector<float>> &input_tensors_py,
      const std::vector<std::vector<int>> &input_shapes,
      // std::vector<std::vector<float>> &output_tensors_py,
      std::vector<nb::ndarray<nb::numpy, float>>
          &output_tensors_py,
      std::vector<std::vector<int>> &output_shapes_out) {

    std::vector<float *> input_tensors(input_tensors_py.size());
    std::vector<float *> output_tensors(output_tensors_py.size());
    // vector<vector<int>> output_shapes(output_tensors.size());
    for (int i = 0; i < input_tensors_py.size(); i++) {
      input_tensors[i] = input_tensors_py[i].data();
    }
    for (int i = 0; i < output_tensors_py.size(); i++) {
      // output_tensors[i] = output_tensors_py[i].data();
      output_tensors[i] = new float[output_tensors_py[i].size()];
      // output_shapes[i] = vector<int>();
    }

    char *param = PyBytes_AsString(param_py.ptr());
    bmcpu_process(bmcpu_handle, op_type, (void *)param, param_size,
                  input_tensors, input_shapes, output_tensors,
                  output_shapes_out);

    for (int i = 0; i < output_tensors_py.size(); i++) {
      // output_tensors[i] = output_tensors_py[i].data();
      memcpy(output_tensors_py[i].data(), output_tensors[i],
             output_tensors_py[i].size() * sizeof(float));
    }

    return output_shapes_out;
  }

  ~PyCpuLayer() { bmcpu_uninit(bmcpu_handle); }

private:
  void *bmcpu_handle;
};

NB_MODULE(pyruntime_bm, m) {
  nb::class_<PythonTensor>(m, "Tensor")
      .def_ro("name", &PythonTensor::name)
      .def_ro("qscale", &PythonTensor::qscale)
      .def_ro("qzero_point", &PythonTensor::qzero_point)
      .def_ro("dtype", &PythonTensor::dtype)
      .def_rw("data", &PythonTensor::data);

  nb::class_<PythonNet>(m, "Net")
      .def("forward", &PythonNet::forward)
      .def("forward_dynamic", &PythonNet::forward_dynamic)
      .def("dump", &PythonNet::dump)
      .def_ro("name", &PythonNet::name)
      .def_ro("num_input", &PythonNet::num_input)
      .def_ro("num_output", &PythonNet::num_output)
      .def_rw("inputs", &PythonNet::inputs)
      .def_rw("outputs", &PythonNet::outputs);

  nb::class_<PythonModel>(m, "Model")
      .def(nb::init<const std::string &, int, const std::string &>(),
           nb::arg("model_file"), nb::arg("device_id") = 0,
           nb::arg("decrypt_lib") = "")
      .def("Net", &PythonModel::Net)
      .def_ro("networks", &PythonModel::networks);

  nb::class_<PyCpuLayer>(m, "CpuLayer")
      .def(nb::init())
      .def("forward", &PyCpuLayer::forward, "forward", nb::arg("op_type"),
           nb::arg("param"), nb::arg("param_size"),
           nb::arg("input_tensors"),
           nb::arg("input_shapes"),
           nb::arg("output_tensors"),
           nb::arg("output_shapes"));
  // Note: nanobind doesn't have scoped_ostream_redirect, we'll handle this differently
}
