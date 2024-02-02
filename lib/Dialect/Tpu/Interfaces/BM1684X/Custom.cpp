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
  values.push_back({0}); // reserved for later dev
  customOpProcessParam(params, values);

  BM168x::call_global_custom_func(api_name.c_str(), values.data(), values.size() * sizeof(custom_param_t),
                                  input_spec->data(), output_spec->data());
}

int64_t tpu::CustomOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto params = getParams();
  vector<custom_param_t> values;
  values.push_back({0}); // reserved for later dev
  customOpProcessParam(params, values);
  int param_size = values.size() * sizeof(custom_param_t);
  if (buffer) {
    char* p = (char*)buffer;
    tpu_param_t info = {0};
    assert(getName().str().size() <= CUSTOM_LAYER_NAME_LEN);
    std::strcpy(info.name, getName().str().c_str());
    info.param_size = param_size;
    info.buffer_addr = -1;
    memcpy(p, &info, sizeof(info));
    p += sizeof(info);
    memcpy(p, values.data(), param_size);
  }
  return sizeof(tpu_param_t) + param_size;
}

int64_t tpu::CustomOp::get_fw_type_bm1684x() { return FW_BMNET_TPU; }
