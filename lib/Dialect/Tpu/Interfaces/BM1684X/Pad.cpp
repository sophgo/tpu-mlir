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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PadOp::codegen_global_bm1684x() {
  std::vector<int64_t> shape_4;
  std::vector<int64_t> pads_4;
  std::vector<int64_t> shape = module::getShape(getInput());
  i64_array_t pads = module::getI64Array(getPaddings());
  auto ret = pad_reset(shape, *pads, shape_4, pads_4);
  if (ret == false) {
    dump();
    llvm_unreachable("Not Implemented");
  }
  std::vector<int64_t> in_shape(shape_4.begin(), shape_4.end());
  std::vector<int64_t> out_shape(shape_4.begin(), shape_4.end());
  pad_param_t param = {0};
  for (int i = 0; i < 4; i++) {
    param.pad[i][0] = pads_4[i];
    param.pad[i][1] = pads_4[i + 4];
    out_shape[i] += (pads_4[i] + pads_4[i + 4]);
  }
  param.type = getMode();
  param.constant = getVal().convertToDouble();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  BM168x::fix_shape(input_spec->at(0), in_shape);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::fix_shape(output_spec->at(0), out_shape);
  BM168x::call_global_func("backend_api_pad_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::PadOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(pad_param_t);
  std::vector<int64_t> shape_4;
  std::vector<int64_t> pads_4;
  std::vector<int64_t> shape = module::getShape(getInput());
  i64_array_t pads = module::getI64Array(getPaddings());
  auto ret = pad_reset(shape, *pads, shape_4, pads_4);
  if (ret == false) {
    dump();
    llvm_unreachable("Not Implemented");
  }
  std::vector<int64_t> in_shape(shape_4.begin(), shape_4.end());
  std::vector<int64_t> out_shape(shape_4.begin(), shape_4.end());
  pad_param_t param = {0};
  for (int i = 0; i < 4; i++) {
    param.pad[i][0] = pads_4[i];
    param.pad[i][1] = pads_4[i + 4];
    out_shape[i] += (pads_4[i] + pads_4[i + 4]);
  }
  param.type = getMode();
  param.constant = getVal().convertToDouble();
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::PadOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice,
                                          group_type_t group_type) {
  llvm_unreachable("Not Implemented");
  return 0;
}

void tpu::PadOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                       group_type_t group_type,
                                       local_sec_info_t &sec_info) {
  llvm_unreachable("Not Implemented");
  return;
}

int64_t tpu::PadOp::get_layer_type() {
  return FW_BMNET_PAD;
}
