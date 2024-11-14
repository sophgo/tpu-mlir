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

void tpu::PackRawOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  pack_raw_spec_t param = {0};
  param.white_level = getWhiteLevel().convertToDouble();
  param.black_level = getBlackLevel().convertToDouble();
  param.threshold = getThreshold().convertToDouble();
  auto channel_order = module::getI64Array(getChannelOrder());
  param.start_point[0] = 0;
  param.start_point[1] = 0;
  for (int i = 0; i < 4; i++) {
    param.channel_order[i] = channel_order->at(i);
  }
  BM168x::call_global_func("backend_api_pack_raw_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

int64_t tpu::PackRawOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pack_raw_spec_t);
  pack_raw_spec_t param{0};
  param.white_level = 4095.f;
  param.black_level = 112.f;
  param.threshold = 1.0f;
  param.channel_order[0] = 1;
  param.channel_order[1] = 0;
  param.channel_order[2] = 2;
  param.channel_order[3] = 3;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::PackRawOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
