//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

typedef struct {
  int int_t;
  float float_t;
  // max size of int and float array is set as 16
  int int_arr_t[16];
  float float_arr_t[16];
} Data;

void processParam(const mlir::ArrayAttr &params, std::vector<Data> &values,
                  int &param_size) {
  for (auto param : params) {
    auto dict = param.dyn_cast<DictionaryAttr>();
    for (auto element : dict) {
      Attribute value_param = element.getValue();
      Data value = {0};
      if (auto int_attr = value_param.dyn_cast<IntegerAttr>()) {
        value.int_t = int_attr.getInt();
        param_size += sizeof(int);
      } else if (auto float_attr = value_param.dyn_cast<FloatAttr>()) {
        value.float_t = float_attr.getValueAsDouble();
        param_size += sizeof(float);
      } else if (auto bool_attr = value_param.dyn_cast<BoolAttr>()) {
        value.int_t = bool_attr.getValue();
        param_size += sizeof(int);
      } else if (auto array_attr = value_param.dyn_cast<ArrayAttr>()) {
        int num = array_attr.size();
        for (int i = 0; i < num; i++) {
          if (auto tmp_value = array_attr[i].dyn_cast<IntegerAttr>()) {
            value.int_arr_t[i] = tmp_value.getInt();
            param_size += sizeof(int);
          } else if (auto tmp_value = array_attr[i].dyn_cast<FloatAttr>()) {
            value.float_arr_t[i] = tmp_value.getValueAsDouble();
            param_size += sizeof(float);
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
  vector<Data> values;
  int param_size = 0;
  processParam(params, values, param_size);

  BM168x::call_global_custom_func(api_name.c_str(), values.data(), param_size,
                                  input_spec->data(), output_spec->data());
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::CustomOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::CustomOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                          int64_t h_step, int64_t d_step,
                                          int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  llvm_unreachable("To be implemented");
}

int64_t tpu::CustomOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::CustomOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
