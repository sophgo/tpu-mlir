//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

// GlobalGenInterface
void tpu::GridSampleInDeformableAttnOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  int interp_mode_int = getInterpMode();
  GridSampleInterpMode interp_mode;
  switch (interp_mode_int) {
  case 0:
    interp_mode = GridSampleBilinear;
    break;
  // case 1:
  //   interp_mode = GridSampleNearest;
  //   break;
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
  grid_sample_in_deformable_attn_global_param_t param = {0};
  param.num_grid_samples = getNumGridSamples();
  param.input_dims = getInputDims();
  for (int i = 0; i < param.num_grid_samples; i++) {
    param.input_global_addr[i] = module::getAddress(getInputGlobalAddr()[i]);
  }
  for (int i = 0; i < param.num_grid_samples; i++) {
    param.grid_global_addr[i] = module::getAddress(getGridGlobalAddr()[i]);
  }
  for (int i = 0; i < param.num_grid_samples; i++) {
    param.attn_global_addr[i] = module::getAddress(getAttnGlobalAddr()[i]);
  }
  param.output_global_addr = module::getAddress(getOutputGlobalAddr());
  param.buffer_global_addr = module::getAddress(getBuffer());
  auto input_n = getInputN();
  for (int i = 0; i < param.num_grid_samples; ++i) {
    auto intAttr = input_n[i].dyn_cast<mlir::IntegerAttr>();
    assert(intAttr && "Expected IntegerAttr");
    param.input_n[i] = static_cast<int>(intAttr.getInt());
  }
  auto input_c = getInputC();
  for (int i = 0; i < param.num_grid_samples; ++i) {
    auto intAttr = input_c[i].dyn_cast<mlir::IntegerAttr>();
    assert(intAttr && "Expected IntegerAttr");
    param.input_c[i] = static_cast<int>(intAttr.getInt());
  }
  auto input_d = getInputD();
  for (int i = 0; i < param.num_grid_samples; ++i) {
    auto intAttr = input_d[i].dyn_cast<mlir::IntegerAttr>();
    assert(intAttr && "Expected IntegerAttr");
    param.input_d[i] = static_cast<int>(intAttr.getInt());
  }
  auto input_h = getInputH();
  for (int i = 0; i < param.num_grid_samples; ++i) {
    auto intAttr = input_h[i].dyn_cast<mlir::IntegerAttr>();
    assert(intAttr && "Expected IntegerAttr");
    param.input_h[i] = static_cast<int>(intAttr.getInt());
  }
  auto input_w = getInputW();
  for (int i = 0; i < param.num_grid_samples; ++i) {
    auto intAttr = input_w[i].dyn_cast<mlir::IntegerAttr>();
    assert(intAttr && "Expected IntegerAttr");
    param.input_w[i] = static_cast<int>(intAttr.getInt());
  }
  param.grid_dout = getGridDout();
  param.grid_hout = getGridHout();
  param.grid_wout = getGridWout();
  param.align_corners = getAlignCorners();
  param.mean = 0;
  param.scale = 0;
  param.need_permute = false;
  param.interp_mode = interp_mode;
  param.padding_mode = padding_mode;
  param.dtype = BM168x::getDataType(getInputGlobalAddr()[0]);

  BM168x::call_global_func("backend_api_grid_sample_in_deformable_attn", &param,
                           sizeof(param));
}

int64_t
tpu::GridSampleInDeformableAttnOp::dyn_codegen_global_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::GridSampleInDeformableAttnOp::get_fw_type_bm1684x() { return -1; }
