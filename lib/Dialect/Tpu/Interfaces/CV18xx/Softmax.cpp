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

void tpu::SoftmaxOp::codegen_global_cv18xx(int64_t layer_id) {
  bool do_log = getLog();
  int axis = this->getAxis();
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t exponential_table_data_lut_gaddr = module::getAddress(getTable());
  gaddr_t exponential_slope_table_data_lut_gaddr =
      module::getAddress(getSlopeTable());
  gaddr_t reciprocal_table_data_lut_gaddr =
      module::getAddress(getReciprocalTable());
  gaddr_t reciprocal_mantissa_table_data_lut_gaddr =
      module::getAddress(getReciprocalMantissaTable());
  std::vector<int64_t> shape = module::getShape(getInput());
  int dimension = shape.size();
  cvi_backend_tg_bf16_softmax_kernel(
      layer_id, ga_input, exponential_table_data_lut_gaddr,
      exponential_slope_table_data_lut_gaddr, reciprocal_table_data_lut_gaddr,
      reciprocal_mantissa_table_data_lut_gaddr, ga_output, shape.data(), axis,
      dimension, do_log);
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::SoftmaxOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::SoftmaxOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                          int64_t d_step, int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info,
                                          int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
