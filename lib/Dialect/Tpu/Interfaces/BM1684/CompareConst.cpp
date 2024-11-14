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

using namespace tpu_mlir::backend;

void tpu::CompareConstOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto binary_type = BM168x::compare_mode(getMode());
  if (!module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getInput()), n * c * h * w,
        getConstVal().convertToDouble(), module::getAddress(getOutput()),
        binary_type, getInversed(), 0, -1.0f,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node,
        module::getStorageType(getInput()).isa<IntegerType>());
  } else {
    int is_signs[3] = {module::isSign(getInput()), 0,
                       module::isSign(getOutput())};
    int is_int8s[3] = {module::getDtypeSize(getInput()) == 1, 1,
                       module::getDtypeSize(getOutput()) == 1};
    int b0_shape[4] = {(int)n, (int)c, (int)h, (int)w};
    BM1684::instance().dl_nodechip_const_binary_fix8b_forward_parallel(
        module::getAddress(getInput()), module::getAddress(getOutput()),
        getConstVal().convertToDouble(), b0_shape, 4, binary_type, 1, 1, 0, 0,
        getInversed(), 0, is_int8s, is_signs,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::CompareConstOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::CompareConstOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                               local_sec_info_t &sec_info) {
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int b0_shape[MAX_SHAPE_DIMS] = {(int)in_g_info.n_slice, (int)c,
                                  (int)in_g_info.h_slice, (int)w};
  auto const_val = (float)getConstVal().convertToDouble();
  auto binary_type = BM168x::compare_mode(getMode());

  if (!module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_const_binary_local(
        in_g_info.out_addr, (uint32_t *)b0_shape, const_val,
        out_g_info.out_addr, binary_type, getInversed(), 0, -1.0f,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else {
    int is_signs[3] = {module::isSign(getInput()), 0,
                       module::isSign(getOutput())};
    int is_int8s[3] = {module::getDtypeSize(getInput()) == 1, 1,
                       module::getDtypeSize(getOutput()) == 1};
    BM1684::instance().dl_nodechip_const_binary_fix8b_forward_local(
        in_g_info.out_addr, out_g_info.out_addr, out_g_info.buffer_addr,
        const_val, b0_shape, 4, binary_type, 1, 1, 0, 0, getInversed(), 0,
        is_int8s, is_signs, BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::CompareConstOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::CompareConstOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::CompareConstOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
