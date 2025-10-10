//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/CustomLayer.h"

// ======================================
// GlobalGenInterface
// ======================================

void tpu::CustomOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  // call backend api according to the custom op name
  std::string op_name = getName().str();
  std::string api_name = "backend_api_" + op_name + "_global";

  // parse param of the custom op
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  *(int64_t *)&values[0] = module::getAddress(getBuffer());
  customOpProcessParam(params, values);

  BM168x::call_global_custom_func(api_name.c_str(), values.data(),
                                  values.size() * sizeof(custom_param_t),
                                  input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CustomOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type,
    bool with_hw_margins) {
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  sec_info.n_slice = in_nslice;
  sec_info.c_slice = in_cslice;
  sec_info.d_slice = in_dslice;
  sec_info.h_slice = in_hslice;
  sec_info.w_slice = in_wslice;
  sec_info.out_n_slice = out_nslice;
  sec_info.out_h_slice = out_hslice;
  sec_info.out_w_slice = out_wslice;

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);

  // call backend api according to the custom op name
  std::string op_name = getName().str();
  std::string api_name = "backend_api_" + op_name + "_local_bfsz";

  // parse param of the custom op
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  return BM168x::call_local_bfsz_custom_func(
      api_name.c_str(), values.data(), values.size() * sizeof(custom_param_t),
      &sec_info, input_spec->data(), output_spec->data());
}

void tpu::CustomOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  auto gi = out_group_infos[0];

  // call backend api according to the custom op name
  std::string op_name = getName().str();
  std::string api_name = "backend_api_" + op_name + "_local";

  // parse param of the custom op
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({.int_t = (int)gi.buffer_addr});
  customOpProcessParam(params, values);
  BM168x::call_local_custom_func(
      api_name.c_str(), values.data(), values.size() * sizeof(custom_param_t),
      &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CustomOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  *(int64_t *)&values[0] = module::getAddress(getBuffer());
  customOpProcessParam(params, values);
  int param_size = values.size() * sizeof(custom_param_t);
  if (buffer) {
    char *p = (char *)buffer;
    tpu_param_t info = {0};
    assert(getName().str().size() <= CUSTOM_LAYER_NAME_LEN);
    std::strcpy(info.name, getName().str().c_str());
    info.param_size = param_size;
    memcpy(p, &info, sizeof(info));
    p += sizeof(info);
    memcpy(p, values.data(), param_size);
  }
  return sizeof(tpu_param_t) + param_size;
}

// ======================================
// Dynamic LocalGenInterface
// ======================================
int64_t tpu::CustomOp::dyn_codegen_local_bm1684x(void *buffer) {
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0});
  *(int64_t *)&values[0] = getGroupInfo(0, 0, 0, 0, 0).buffer_addr;
  customOpProcessParam(params, values);
  int param_size = values.size() * sizeof(custom_param_t);
  if (buffer) {
    char *p = (char *)buffer;
    tpu_param_t info = {0};
    assert(getName().str().size() <= CUSTOM_LAYER_NAME_LEN);
    std::strcpy(info.name, getName().str().c_str());
    info.param_size = param_size;
    memcpy(p, &info, sizeof(info));
    p += sizeof(info);
    memcpy(p, values.data(), param_size);
  }
  return sizeof(tpu_param_t) + param_size;
}

int64_t tpu::CustomOp::get_fw_type_bm1684x() { return FW_BMNET_TPU; }
