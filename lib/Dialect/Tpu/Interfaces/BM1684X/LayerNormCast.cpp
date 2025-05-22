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

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::LayerNormCastOp::codegen_global_bm1684x() {
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());

  int qInput = module::isUniformQuantized(getInput());
  bool fOutput = out_type.isIntOrIndex() == false;

  bool fInput = in_type.isIntOrIndex() == false;
  bool qOutput = module::isUniformQuantized(getOutput());

  auto round_mode = round_mode_convert(getRoundMode());
  auto op1 = getOperation();
  op1 = op1;
  layer_norm_cast_global_spec_t param = {0};
  if (fInput && qOutput) {
    param.isCastAtEnd = getIsCastAtEnd();
    assert(param.isCastAtEnd == 1); // Not support CastLayerNorm

    // layernorm
    const bool have_weight = !getWeight().getType().isa<NoneType>();
    const bool have_bias = !getBias().getType().isa<NoneType>();

    param.common_layer_norm.axis = (int)getAxis();
    param.common_layer_norm.eps = getEps().convertToDouble();
    param.common_layer_norm.affine = (have_weight << 0) + (have_bias << 1);
    param.common_layer_norm.need_mean = false;
    param.common_layer_norm.need_rstd = false;

    // requant
    auto qtype = module::getUniformQuantizedType(getOutput());
    param.common_dequant.scale_value = 1.0 / qtype.getScale();
    param.common_dequant.offset_value = qtype.getZeroPoint();
    param.common_dequant.round_mode = round_mode;
  } else if (qInput && fOutput) {
    param.isCastAtEnd = getIsCastAtEnd();

    // layernorm
    const bool have_weight = !getWeight().getType().isa<NoneType>();
    const bool have_bias = !getBias().getType().isa<NoneType>();

    param.common_layer_norm.axis = (int)getAxis();
    param.common_layer_norm.eps = getEps().convertToDouble();
    param.common_layer_norm.affine = (have_weight << 0) + (have_bias << 1);
    param.common_layer_norm.need_mean = false;
    param.common_layer_norm.need_rstd = false;

    // dequant
    auto qtype = module::getUniformQuantizedType(getInput());
    param.common_dequant.scale_value = qtype.getScale();
    param.common_dequant.offset_value = qtype.getZeroPoint();
    param.common_dequant.round_mode = round_mode;
  } else {
    UNREACHABLE_THIS(0);
  }
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_layer_norm_cast_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LayerNormCastOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::LayerNormCastOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
