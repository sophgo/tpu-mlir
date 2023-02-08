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
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::SwapDimInnerOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  swap_dim_spec_t param = {0};
  auto offset = module::getI64Array(getOffset());
  for (int i = 0; i < offset->size(); ++i) {
    if (offset->at(i) != 0) {
      param.axis_list[param.axis_num] = i;
      param.offset_list[param.axis_num] = offset->at(i);
      param.axis_num += 1;
    }
  }

  BM168x::call_global_func("backend_api_swap_dim_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

int64_t tpu::SwapDimInnerOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}
// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SwapDimInnerOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice,
    group_type_t group_type) {
  return 0;
}

void tpu::SwapDimInnerOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                                group_type_t group_type,
                                                local_sec_info_t &sec_info) {
  llvm_unreachable("Not Implemented");
}
