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

void tpu::ShuffleChannelOp::codegen_global_bm1684x() {
  shuffle_channel_param_t spec = {0};
  spec.group = getGroup();
  auto op = getOperation();
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  module::getNCHW(getInput(), in, ic, ih, iw, false);
  module::getNCHW(getOutput(), on, oc, oh, ow, false);
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::fix_shape(input_spec->at(0), {in, ic, ih, iw});
  BM168x::fix_shape(output_spec->at(0), {on, oc, oh, ow});
  BM168x::call_global_func("backend_api_channel_shuffle_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShuffleChannelOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::ShuffleChannelOp::get_fw_type_bm1684x() {
  return FW_LAYER_UNKNOWN;
}
