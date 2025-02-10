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

// ======================================
// GlobalGenInterface
// ======================================

void tpu::ConvbwdOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  Convbwd_param_t spec;
  memset(&spec, 0, sizeof(spec));
  auto &common = spec.common;
  // multi-core backend requires L2 and sdma which only available for bm1690
  spec.use_multi_core = 0;
  if (module::isBM1690Family()) {
    spec.use_multi_core = 1;
  }
  common.groups = attr.groups;
  common.n = attr.n;
  common.ic = attr.ic;
  common.ih = attr.ih;
  common.iw = attr.iw;
  common.oc = attr.oc;
  common.oh = attr.oh;
  common.ow = attr.ow;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.sh = attr.sh;
  common.sw = attr.sw;
  common.pt = attr.pht;
  common.pb = attr.phb;
  common.pl = attr.pwl;
  common.pr = attr.pwr;
  common.insh = attr.insh;
  common.insw = attr.insw;
  bool buffer_enable = getGradInputEnable() || getGradWeightEnable();
  spec.buffer_addr = buffer_enable ? module::getAddress(getBuffer()) : 0;
  spec.grad_input_enable = getGradInputEnable();
  spec.grad_weight_enable = getGradWeightEnable();
  spec.grad_bias_enable = getGradBiasEnable();
  BM168x::call_global_func("backend_api_conv_bwd_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

void tpu::ConvbwdOp::codegen_global_bm1684() {}

void tpu::ConvbwdOp::codegen_global_cv18xx(int64_t layer_id) {}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ConvbwdOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::ConvbwdOp::get_fw_type_bm1684x() { return 0; }
