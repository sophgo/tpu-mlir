//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

typedef union {
  int int_t;
  float float_t;
  // max size of int and float array is set as 16
  int int_arr_t[16];
  float float_arr_t[16];
} custom_param_t;

#define CUSTOM_LAYER_NAME_LEN 20
typedef struct {
  char     name[CUSTOM_LAYER_NAME_LEN + 1];
  int      param_size;
  uint64_t buffer_addr;
  int      buffer_size;
} tpu_param_t;

static void processParam(const mlir::ArrayAttr &params, std::vector<custom_param_t> &values) {
  for (auto param : params) {
    auto dict = param.dyn_cast<DictionaryAttr>();
    for (auto element : dict) {
      Attribute value_param = element.getValue();
      custom_param_t value = {0};
      if (auto int_attr = value_param.dyn_cast<IntegerAttr>()) {
        value.int_t = int_attr.getInt();
      } else if (auto float_attr = value_param.dyn_cast<FloatAttr>()) {
        value.float_t = float_attr.getValueAsDouble();
      } else if (auto bool_attr = value_param.dyn_cast<BoolAttr>()) {
        value.int_t = bool_attr.getValue();
      } else if (auto array_attr = value_param.dyn_cast<ArrayAttr>()) {
        int num = array_attr.size();
        for (int i = 0; i < num; i++) {
          if (auto tmp_value = array_attr[i].dyn_cast<IntegerAttr>()) {
            value.int_arr_t[i] = tmp_value.getInt();
          } else if (auto tmp_value = array_attr[i].dyn_cast<FloatAttr>()) {
            value.float_arr_t[i] = tmp_value.getValueAsDouble();
          } else {
            llvm_unreachable("Only int and float vector supported now");
          }
        }
      } else {
        llvm_unreachable("Type of parameter unsupported");
      }
      values.push_back(value);
    }
  }
}

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
  processParam(params, values);

  BM168x::call_global_custom_func(api_name.c_str(), values.data(), values.size() * sizeof(custom_param_t),
                                  input_spec->data(), output_spec->data());
}

int64_t tpu::CustomOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto params = getParams();
  vector<custom_param_t> values;
  processParam(params, values);
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
