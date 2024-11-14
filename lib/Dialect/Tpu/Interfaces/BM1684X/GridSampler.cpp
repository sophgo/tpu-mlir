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

// =========================================
// GlobalGenInterface
// =========================================

void tpu::GridSamplerOp::codegen_global_bm1684x() {
  grid_sample_global_param_t param = {0};
  memset(&param, 0, sizeof(param));
  param.input_addr = module::getAddress(getInput());
  param.grid_addr = module::getAddress(getGrid());
  param.output_addr = module::getAddress(getOutput());
  param.buffer_addr = module::getAddress(getBuffer());

  param.align_corners = getAlignCorners();
  // param.interp_mode = GridSampleNearest;
  int interp_mode_int = getMode();
  GridSampleInterpMode interp_mode;
  switch (interp_mode_int) {
  case 0:
    interp_mode = GridSampleBilinear;
    break;
  case 1:
    interp_mode = GridSampleNearest;
    break;
  default:
    llvm_unreachable("not implemented.");
    break;
  }
  int padding_mode_int = getPaddingMode();
  GridSamplePaddingMode padding_mode;
  switch (padding_mode_int) {
  case 0:
    padding_mode = GridSampleZeros;
    break;
  case 1:
    padding_mode = GridSampleBorder;
    break;
  case 2:
    padding_mode = GridSampleReflection;
    break;
  default:
    llvm_unreachable("not implemented.");
    break;
  }
  param.interp_mode = interp_mode;
  param.padding_mode = padding_mode;
  param.mean = getMean().convertToDouble();
  param.scale = getScale().convertToDouble();
  param.need_permute = getNeedPermute();

  auto in_shape = module::getShape(getInput());
  auto out_shape = module::getShape(getOutput());
  auto dims = in_shape.size();
  param.dims = dims;

  if (dims == 4) {
    param.input_n = in_shape[0];
    param.input_c = in_shape[1];
    param.input_h = in_shape[2];
    param.input_w = in_shape[3];
    param.output_h = out_shape[2];
    param.output_w = out_shape[3];
  } else if (dims == 5) {
    param.input_n = in_shape[0];
    param.input_c = in_shape[1];
    param.input_d = in_shape[2];
    param.input_h = in_shape[3];
    param.input_w = in_shape[4];
    param.output_d = out_shape[2];
    param.output_h = out_shape[3];
    param.output_w = out_shape[4];
  } else {
    llvm_unreachable("Not implemented.");
  }

  param.dtype = BM168x::getDataType(getInput());
  if (support_multi_core() && interp_mode == GridSampleBilinear) {
    BM168x::call_global_func("backend_api_grid_sample_multi_core_global",
                             &param, sizeof(grid_sample_global_param_t));
    return;
  }
  BM168x::call_global_func("backend_api_grid_sample_global", &param,
                           sizeof(grid_sample_global_param_t));
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GridSamplerOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(grid_sample_global_param_t);
  grid_sample_global_param_t param = {0};
  memset(&param, 0, sizeof(param));
  param.input_addr = module::getAddress(getInput());
  param.grid_addr = module::getAddress(getGrid());
  param.output_addr = module::getAddress(getOutput());
  param.buffer_addr = module::getAddress(getBuffer());

  param.align_corners = getAlignCorners();
  int interp_mode_int = getMode();
  GridSampleInterpMode interp_mode;
  switch (interp_mode_int) {
  case 0:
    interp_mode = GridSampleBilinear;
    break;
  case 1:
    interp_mode = GridSampleNearest;
    break;
  default:
    llvm_unreachable("not implemented.");
    break;
  }
  int padding_mode_int = getPaddingMode();
  GridSamplePaddingMode padding_mode;
  switch (padding_mode_int) {
  case 0:
    padding_mode = GridSampleZeros;
    break;
  case 1:
    padding_mode = GridSampleBorder;
    break;
  case 2:
    padding_mode = GridSampleReflection;
    break;
  default:
    llvm_unreachable("not implemented.");
    break;
  }
  param.interp_mode = interp_mode;
  param.padding_mode = padding_mode;
  param.mean = getMean().convertToDouble();
  param.scale = getScale().convertToDouble();
  param.need_permute = getNeedPermute();

  auto in_shape = module::getShape(getInput());
  auto out_shape = module::getShape(getOutput());
  auto dims = in_shape.size();
  param.dims = dims;

  if (dims == 4) {
    param.input_n = in_shape[0];
    param.input_c = in_shape[1];
    param.input_h = in_shape[2];
    param.input_w = in_shape[3];
    param.output_h = out_shape[2];
    param.output_w = out_shape[3];
  } else if (dims == 5) {
    param.input_n = in_shape[0];
    param.input_c = in_shape[1];
    param.input_d = in_shape[2];
    param.input_h = in_shape[3];
    param.input_w = in_shape[4];
    param.output_d = out_shape[2];
    param.output_h = out_shape[3];
    param.output_w = out_shape[4];
  } else {
    llvm_unreachable("Not implemented.");
  }

  param.dtype = BM168x::getDataType(getInput());

  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic LocalGenInterface
// ======================================

int64_t tpu::GridSamplerOp::get_fw_type_bm1684x() {
  return FW_LAYER_GRIDSAMPLER;
}
