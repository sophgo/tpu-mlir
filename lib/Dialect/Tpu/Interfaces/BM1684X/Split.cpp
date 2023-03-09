//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

void tpu::SplitOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  std::vector<int64_t> input_shape = module::getShape(getInput());
  split_spec_t param = {0};
  param.input_num = 1;
  std::vector<uint64_t> output_addr;
  for (int i = 0; i < getNum(); ++i) {
    output_addr.push_back(module::getAddress(getOutputs()[i]));
  }
  param.axis = getAxis();
  param.split_num = getNum();
  int max_split_num = (input_shape[getAxis()] + getNum() - 1) / getNum();
  std::vector<int> split_size;
  for (int i = 0; i < getNum() - 1; ++i) {
    param.split_size[i] = max_split_num;
  }
  param.split_size[getNum() - 1] =
      input_shape[getAxis()] - max_split_num * (getNum() - 1);
  BM168x::call_global_func("backend_api_split_tf_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SplitOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(split_spec_t);
  std::vector<int64_t> input_shape = module::getShape(getInput());
  split_spec_t param = {0};
  param.input_num = 1;
  std::vector<uint64_t> output_addr;
  for (int i = 0; i < getNum(); ++i) {
    output_addr.push_back(module::getAddress(getOutputs()[i]));
  }
  param.axis = getAxis();
  param.split_num = getNum();
  int max_split_num = (input_shape[getAxis()] + getNum() - 1) / getNum();
  std::vector<int> split_size;
  for (int i = 0; i < getNum() - 1; ++i) {
    param.split_size[i] = max_split_num;
  }
  param.split_size[getNum() - 1] =
      input_shape[getAxis()] - max_split_num * (getNum() - 1);
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::SplitOp::get_fw_type_bm1684x() {
  return FW_BMNET_SPLIT_TF;
}
