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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;


// =========================================
// GlobalGenInterface
// =========================================
void tpu::InterpOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  interp_global_spec_t param = {0};
  auto &common = param.common;
  common.pad_bag = 0;
  common.pad_end = 0;
  int coord = 0;
  if (getCoordMode() == tpu::ResizeCoordMode::half_pixel)
    coord = 0;
  else if (getCoordMode() == tpu::ResizeCoordMode::pytorch_half_pixel)
    coord = 1;
  else if (getCoordMode() == tpu::ResizeCoordMode::align_corners)
    coord = 2;
  if (getMode() == tpu::ResizeMode::nearest) {
    common.platform_sp = ONNX_NEAREST;
    common.align_corners = true;
    common.half_pixel_centers = false;
  } else if (getMode() == tpu::ResizeMode::linear) {
    common.platform_sp = PYTORCH_SUPPORT;
    common.align_corners = (coord == 2) ? 1 : 0;
    common.half_pixel_centers = (coord == 0 || coord == 1) ? 1 : 0;
  }
  BM168x::call_global_func("backend_api_interp_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}
#if 0
// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::InterpOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice, group_type_t group_type) {
  return 0;
}

void tpu::InterpOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  int64_t n, c, ih, iw, oh, ow;
  module::getNCHW(getInput(), n, c, ih, iw);
  module::getNCHW(getOutput(), n, c, oh, ow);
  interp_local_param_t param = {0};
  param.input_addr = in_gi.out_addr;
  param.output_addr = gi.out_addr;
  param.input_n = static_cast<int32_t>(in_gi.n_slice);
  param.input_c = static_cast<int32_t>(c);
  param.input_h = static_cast<int32_t>(in_gi.h_slice);
  param.input_w = static_cast<int32_t>(iw);
  param.output_h = gi.h_slice;
  param.output_w = ow;
  param.pad_bag = 0;
  param.pad_end = 0;
  param.dtype = BM168x::getDataType(getInput());

  int coord = 0;
  bool align_corners = (getCoordMode() == tpu::ResizeCoordMode::align_corners);
  bool half_pixel = (getCoordMode() == tpu::ResizeCoordMode::half_pixel);
  if (getCoordMode() == tpu::ResizeCoordMode::half_pixel)
    coord = 0;
  else if (getCoordMode() == tpu::ResizeCoordMode::pytorch_half_pixel)
    coord = 1;
  else if (getCoordMode() == tpu::ResizeCoordMode::align_corners)
    coord = 2;
  if (getMode() == tpu::ResizeMode::nearest) {
    param.platform_sp = ONNX_NEAREST;
    param.align_corners = true;
    param.half_pixel_centers = false;
  } else if (getMode() == tpu::ResizeMode::linear) {
    param.platform_sp = PYTORCH_SUPPORT;
    param.align_corners = (coord == 2) ? 1: 0;
    param.half_pixel_centers = (coord == 0 || coord == 1) ? 1 : 0;
  }
  auto op = getOperation();
  BM168x::call_local_func("backend_api_interp_local", &param,
                                      sizeof(param));
}
#endif

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::InterpOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(interp_global_spec_t);
  interp_global_spec_t param = {0};
  auto &common = param.common;
  common.pad_bag = 0;
  common.pad_end = 0;
  int coord = 0;
  if (getCoordMode() == tpu::ResizeCoordMode::half_pixel)
    coord = 0;
  else if (getCoordMode() == tpu::ResizeCoordMode::pytorch_half_pixel)
    coord = 1;
  else if (getCoordMode() == tpu::ResizeCoordMode::align_corners)
    coord = 2;
  if (getMode() == tpu::ResizeMode::nearest) {
    common.platform_sp = ONNX_NEAREST;
    common.align_corners = true;
    common.half_pixel_centers = false;
  } else if (getMode() == tpu::ResizeMode::linear) {
    common.platform_sp = PYTORCH_SUPPORT;
    common.align_corners = (coord == 2) ? 1 : 0;
    common.half_pixel_centers = (coord == 0 || coord == 1) ? 1 : 0;
  }
  param.shape_is_fixed = true;
  auto out_shape = module::getShape(getOutput());
  param.dims = out_shape.size();
  for (int i=0; i < param.dims; i++)
    param.shape[i] = out_shape[i];
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::InterpOp::get_fw_type_bm1684x() {
  return FW_BMNET_INTERP;
}
