//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::ActiveOp::codegen_global_bm1684() {
  // active_global_spec_t
  uint64_t bottom_global_offset = module::getAddress(getInput());
  uint64_t top_global_offset = module::getAddress(getResult());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  uint64_t length = n * c * h * w;
  int activate_type = (int)getMode();
  if (activate_type != (int)ActiveMode::SIGMOID &&
      activate_type != (int)ActiveMode::EXP) {
    llvm_unreachable("Not Implement such activate type!");
  }
  float coeffs[8];
  if (getCoeffs().has_value()) {
    const auto coeffs_ = module::getF64Array(getCoeffs().value());
    for (int i = 0; i < coeffs_->size(); ++i) {
      coeffs[i] = (float)coeffs_->at(i);
    }
  }
  if (module::isUniformQuantized(getOutput())) {
    llvm_unreachable("Not Implemented!");
  } else {
    BM1684::instance().dl_nodechip_active_forward_parallel(
        bottom_global_offset, top_global_offset, length, activate_type, coeffs,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}

int64_t tpu::ActiveOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  int64_t tensor_size = in_lmem_bytes / in_nslice;
  if (!module::isUniformQuantized(getOutput())) {
    switch (getMode()) {
    case ActiveMode::SIGMOID:
    case ActiveMode::SILU:
    case ActiveMode::EXP:
      buffer_size = tensor_size;
      break;
    default:
      llvm_unreachable("Not Implement such activate type");
      break;
    }
  }

  return buffer_size;
}

void tpu::ActiveOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  llvm_unreachable("ActiveOp Local layer has bugs");
  auto out_g_info = getGroupInfo(n_step, h_step);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  uint32_t bottom_dim[MAX_SHAPE_DIMS] = {
      (uint32_t)in_g_info.n_slice, (uint32_t)c, (uint32_t)in_g_info.h_slice,
      (uint32_t)w};
  uint32_t bottom_local_offset = in_g_info.out_addr;
  uint32_t top_local_offset = out_g_info.out_addr;
  uint32_t imm_buffer_local_offet = out_g_info.buffer_addr;
  int depth = 1; // TODO: makesure layer_group == NORMAL
  int activate_type = (int)getMode();

  if (module::isUniformQuantized(getOutput())) {
    llvm_unreachable("Not support local fix8b layer");
  } else {
    float prelu_slope = 0.0;
    BM1684::instance().dl_nodechip_active_forward_local(
        bottom_local_offset, top_local_offset, imm_buffer_local_offet,
        bottom_dim[0] * depth, bottom_dim[1], bottom_dim[2], bottom_dim[3],
        activate_type, &prelu_slope,
        (CMD_ID_NODE *)BM1684::instance().bdc_node);
  }
}
