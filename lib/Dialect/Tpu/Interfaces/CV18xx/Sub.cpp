//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::SubOp::codegen_global_cv18xx(int64_t layer_id) {
  int input_num = getInputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w, bn, bc, bh, bw;
  module::getNCHW(getInputs()[0], n, c, h, w, false);
  module::getNCHW(getInputs()[1], bn, bc, bh, bw, false);
  std::vector<gaddr_t> ga_inputs;
  gaddr_t ga_a = module::getAddress(getInputs()[0]);
  gaddr_t ga_b = module::getAddress(getInputs()[1]);
  gaddr_t ga_output = module::getAddress(getOutput());

  if (module::isUniformQuantized(getOutput())) {
    auto multiplier_v = module::getI64Array(getMultipliers(), input_num, 1);
    auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
    std::vector<int32_t> multiplier;
    multiplier.assign(multiplier_v->begin(), multiplier_v->end());

    cvi_backend_tg_int8_bcast_sub_kernel(
        layer_id, ga_a, ga_b, ga_output, n, c, h, w, bn, bc, bh, bw,
        getDoRelu(), (int32_t)rshift_v->at(0), multiplier.data());

  } else {
    cvi_backend_tg_bf16_bcast_sub_kernel(layer_id, ga_a, ga_b, ga_output, n, c,
                                         h, w, bn, bc, bh, bw, getDoRelu());
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SubOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::SubOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                      int64_t d_step, int64_t w_step,
                                      group_type_t group_type,
                                      local_sec_info_t &sec_info,
                                      int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
