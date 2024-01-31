//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/CustomLayer.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

LogicalResult tpu::CustomOp::init(InferenceParameter &p) { return success(); }
void tpu::CustomOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CustomOp::inference(InferenceParameter &p) {
  const int num_input = getInputs().size();
  std::vector<int[MAX_SHAPE_DIMS]> in_shapes_v(num_input);
  std::vector<int> in_dims_v(num_input);
  for (int i = 0; i < num_input; ++i) {
    auto shape = module::getShape(getInputs()[i]);
    assert(shape.size() <= MAX_SHAPE_DIMS);
    in_dims_v[i] = shape.size();
    for (int j = 0; j < shape.size(); j++) {
      in_shapes_v[i][j] = shape[j];
    }
  }
  auto params = getParams();
  std::vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  std::string op_name = getName().str();
  std::string api_name = "inference_" + op_name;
  void* args[4] = {(void*)in_shapes_v.data(), (void*)in_dims_v.data(),
                   (void*)p.inputs.data(), (void*)p.outputs.data()};
  bool ret = false;
  BM168x::call_custom_plugin_func(
    kCustomPluginTypes::PLUGIN_INFERENCE, &ret,
    api_name.c_str(), values.data(),
    values.size() * sizeof(custom_param_t),
    args);
  if (ret) return success();
  else return failure();
}

mlir::Type tpu::CustomOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

LogicalResult tpu::CustomOp::LocalGenSupport() {
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  std::string op_name = getName().str();
  std::string api_name = "local_gen_support_" + op_name;
  bool ret = false;
  BM168x::call_custom_plugin_func(
    kCustomPluginTypes::PLUGIN_LOCALGENSUPPORT, &ret,
    api_name.c_str(), values.data(),
    values.size() * sizeof(custom_param_t),
    nullptr);
  if (ret) return success();
  else return failure();
}

LogicalResult tpu::CustomOp::AllowDataSplit(int64_t axis, group_type_t group_type) {
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  std::string op_name = getName().str();
  std::string api_name = "allow_data_split_" + op_name;
  bool ret = false;
  int args[2] = {(int)axis, (int)group_type};
  BM168x::call_custom_plugin_func(
    kCustomPluginTypes::PLUGIN_ALLOWDATASPLIT, &ret,
    api_name.c_str(), values.data(),
    values.size() * sizeof(custom_param_t),
    args);
  if (ret) return success();
  else return failure();
}

LogicalResult tpu::CustomOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  std::string op_name = getName().str();
  std::string api_name = "backward_h_" + op_name;
  bool ret = false;
  int args[4] = {-1, -1, (int)out_idx, (int)out_slice};
  BM168x::call_custom_plugin_func(
    kCustomPluginTypes::PLUGIN_BACKWARDH, &ret,
    api_name.c_str(), values.data(),
    values.size() * sizeof(custom_param_t),
    args);
  in_idx = args[0];
  in_slice = args[1];
  if (ret) return success();
  else return failure();
}

LogicalResult tpu::CustomOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  std::string op_name = getName().str();
  std::string api_name = "backward_w_" + op_name;
  bool ret = false;
  int args[4] = {-1, -1, (int)out_idx, (int)out_slice};
  BM168x::call_custom_plugin_func(
    kCustomPluginTypes::PLUGIN_BACKWARDW, &ret,
    api_name.c_str(), values.data(),
    values.size() * sizeof(custom_param_t),
    args);
  in_idx = args[0];
  in_slice = args[1];
  if (ret) return success();
  else return failure();
}
