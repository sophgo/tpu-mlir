//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::Conv3DOp::codegen_global_cv18xx(int64_t layer_id) {
  if (module::isUniformQuantized(getOutput())) {
    llvm_unreachable("Not supported now");
  }
  auto attr = parseParam();
  if (attr.groups != 1) {
    llvm_unreachable("Not supported now");
  }
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_filter = module::getAddress(getFilter());
  gaddr_t ga_bias = GA_INVALID;
  if (attr.has_bias) {
    ga_bias = module::getAddress(getBias());
  }

  cvi_backend_tg_bf16_conv3d_kernel(
      layer_id,  // layer_id
      ga_input,  // input_data_gaddr,
      ga_output, // output_data_gaddr,
      ga_filter, // weight_data_gaddr,
      ga_bias,   // bias_data_gaddr,
      attr.n, attr.ic, attr.id, attr.ih, attr.iw, attr.oc, attr.od, attr.oh,
      attr.ow, attr.kd, attr.kh, attr.kw, attr.dd, attr.dh, attr.dw, attr.pdf,
      attr.pdb, attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sd, attr.sh,
      attr.sw,
      attr.has_bias,       // bias_term,
      attr.do_relu ? 1 : 0 // do_activation,
  );
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv3DOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::Conv3DOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info,
                                         int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
