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

void tpu::MeanStdScaleOp::codegen_global_bm1684x() {
  auto std = module::getF64Array(getStd());
  auto scale = module::getF64Array(getScale());
  auto mean = module::getF64Array(getMean());
  auto zero_points = module::getF64Array(getZeroPoints());
  auto round_mode = round_mode_convert(symbolizeRoundMode(getRoundingMode()).value());
  auto rshift = module::getI32Array(getRshift());
  auto offset = module::getI32Array(getOffset());
  auto multi = module::getI32Array(getMulti());
  std::vector<int64_t> in_shape = module::getShape(getInput());

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  mean_std_scale_param_t param = {0};
  param.num_of_chn = in_shape[1];
  memcpy(param.std, std->data(), sizeof(param.std));
  memcpy(param.mean, mean->data(), sizeof(param.mean));
  memcpy(param.scale, scale->data(), sizeof(param.scale));
  param.in_zp = zero_points->at(0);
  param.out_zp = zero_points->at(1);
  memcpy(param.multi , multi->data(), sizeof(param.multi));
  memcpy(param.rshift , rshift->data(), sizeof(param.rshift));
  memcpy(param.offset , offset->data(), sizeof(param.offset));
  param.round_mode = round_mode;

  BM168x::call_global_func("backend_api_mean_std_scale_global", &param,
          sizeof(param), input_spec->data(), output_spec->data());
}

int64_t tpu::MeanStdScaleOp::dyn_codegen_global_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::MeanStdScaleOp::get_fw_type_bm1684x() {
  return FW_LAYER_UNKNOWN;
}

void tpu::MeanStdScaleOp::codegen_global_bm1684() {
}

uint32_t tpu::MeanStdScaleOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::MeanStdScaleOp::get_fw_type_bm1684() {
  return FW_LAYER_UNKNOWN;
}

void tpu::MeanStdScaleOp::codegen_global_cv18xx(int64_t layer_id) {
}
