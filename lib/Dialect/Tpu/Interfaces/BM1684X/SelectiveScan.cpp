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

// typedef struct  {
//    uint64_t output_addr;
//    uint64_t Cs_addr;
//    uint64_t deltaA_addr;
//    uint64_t deltaB_u_addr;
//    uint64_t us_addr;
//    uint64_t Ds_addr;
//    int Batch;
//    int KC_dim;
//    int L;
//    int dtype;
// } selective_scan_common_spec_t;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::SelectiveScanOp::codegen_global_bm1684x() {
  selective_scan_common_spec_t p = {0};
  auto Cs_shape = module::getShape(getCs());

  p.output_addr = module::getAddress(getOutput());
  p.Cs_addr = module::getAddress(getCs());
  p.deltaA_addr = module::getAddress(getDeltaA());
  p.deltaB_u_addr = module::getAddress(getDeltaBU());
  p.us_addr = module::getAddress(getUs());
  p.Ds_addr = module::getAddress(getDs());
  p.Batch = Cs_shape[3];
  p.KC_dim = Cs_shape[1] / 2;
  p.L = Cs_shape[0];
  p.dtype = BM168x::getDataType(getOutput());

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  BM168x::call_ppl_global_func("api_selective_scan_global", &p,
                               sizeof(selective_scan_common_spec_t),
                               input_spec->data(), output_spec->data());
}

int64_t tpu::SelectiveScanOp::dyn_codegen_global_bm1684x(void *buffer) {
  llvm_unreachable("Not supported now");
  return 0;
}

int64_t tpu::SelectiveScanOp::get_fw_type_bm1684x() {
  return FW_BMNET_SELECTIVESCAN;
}
