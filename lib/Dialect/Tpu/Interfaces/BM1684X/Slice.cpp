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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
using namespace tpu_mlir::backend;

void tpu::SliceOp::codegen_global_bm1684x() {
  auto p = parseParam();
  if (p.fusible) {
    return;
  }
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::fix_shape(input_spec->at(0), p.is_4);
  BM168x::fix_shape(output_spec->at(0), p.os_4);
  strideslice_common_spec_t param = {0};
  param.begin_mask = 0;
  param.end_mask = 0;
  int num_dims = p.is_4.size();
  for (int i = 0; i < num_dims; i++) {
    param.begin_index[i] = p.offset_4[i];
    param.end_index[i] = p.os_4[i] * p.step_4[i] + p.offset_4[i];
    param.strides[i] = p.step_4[i];
  }
  BM168x::call_global_func("backend_api_strideslice_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SliceOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  strideslice_local_spec_t spec = {0};
  auto &common = spec.common;
  common.begin_mask = 0;
  common.end_mask = 0;
  auto output_shape = SmallVector<int64_t>(module::getShape(getOutput()));
  const int num_dims = output_shape.size();
  output_shape[0] = out_nslice;
  if (num_dims > 2) {
    output_shape[2] = out_hslice;
  }
  if (num_dims > 3) {
    output_shape[3] = out_wslice;
  }
  const auto offset = module::getI64Array(getOffset());
  const auto steps = module::getI64Array(getSteps());
  for (int i = 0; i < num_dims; i++) {
    common.begin_index[i] = offset->at(i);
    common.strides[i] = steps->at(i);
    common.end_index[i] =
        common.begin_index[i] + output_shape[i] * common.strides[i];
  }
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  // int64_t n, c, d, h, w, on, oc, od, oh, ow;
  // auto input = op->getOperand(0);
  // auto output = op->getResult(0);
  // module::getNCDHW(input, n, c, d, h, w, group_type);
  // module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  sec_info.n_slice = in_nslice;
  sec_info.d_slice = in_dslice;
  sec_info.h_slice = in_hslice;
  sec_info.w_slice = in_wslice;
  sec_info.out_n_slice = out_nslice;
  sec_info.out_h_slice = out_hslice;
  sec_info.out_w_slice = out_wslice;
  return BM168x::call_local_bfsz_func("backend_api_strideslice_local_bfsz", &spec, sizeof(spec),
                                      &sec_info, input_spec->data(), output_spec->data());
}

void tpu::SliceOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  strideslice_local_spec_t spec = {0};
  const auto &gi = getGroupInfo(0, 0, 0, 0);
  spec.buffer_addr = gi.buffer_addr;
  auto &common = spec.common;
  common.begin_mask = 0;
  common.end_mask = 0;
  auto output_shape = SmallVector<int64_t>(module::getShape(getOutput()));
  const int num_dims = output_shape.size();
  output_shape[0] = sec_info.out_n_slice;
  if (num_dims > 2) {
    output_shape[2] = sec_info.out_h_slice;
  }
  if (num_dims > 3) {
    output_shape[3] = sec_info.out_w_slice;
  }
  const auto offset = module::getI64Array(getOffset());
  const auto steps = module::getI64Array(getSteps());
  for (int i = 0; i < num_dims; i++) {
    common.begin_index[i] = offset->at(i);
    common.strides[i] = steps->at(i);
    common.end_index[i] =
        common.begin_index[i] + output_shape[i] * common.strides[i];
  }

  BM168x::call_local_func("backend_api_strideslice_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SliceOp::dyn_codegen_global_bm1684x(void *buffer) {
#if 0
  if (!buffer)
    return sizeof(strideslice_common_spec_t);
  strideslice_common_spec_t param = {0};
  std::vector<int64_t> input_shape = module::getShape(getInput());
  std::vector<int64_t> output_shape = module::getShape(getOutput());
  param.begin_mask = 0;
  param.end_mask = 0;
  int num_dims = input_shape.size();
  auto&& offset = getOffset();
  auto&& step = getSteps();
  for (int i = 0; i < num_dims; i++) {
    param.begin_index[i] = offset[i].cast<IntegerAttr>().getInt();
    param.end_index[i] = output_shape[i] * step[i].cast<IntegerAttr>().getInt()
                  + offset[i].cast<IntegerAttr>().getInt();
    param.strides[i] = step[i].cast<IntegerAttr>().getInt();
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
#endif
  return 0;
}

int64_t tpu::SliceOp::get_fw_type_bm1684x() {
  // return FW_BMNET_SLICE;
  return FW_BMNET_STRIDESLICE;
}
