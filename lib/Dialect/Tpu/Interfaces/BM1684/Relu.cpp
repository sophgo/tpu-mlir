//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::ReluOp::codegen_global_bm1684() {
  uint64_t bottom_global_offset = module::getAddress(getInput());
  uint64_t top_global_offset = module::getAddress(getResult());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (!module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_relu_forward_32bit_parallel(
        bottom_global_offset, top_global_offset, 0.0f,
        (float)getReluLimit().convertToDouble(), n, c, h, w,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_prelu_forward_fix8b(
        bottom_global_offset,
        /*slope addr*/ 0, top_global_offset, /*slop*/ 0, /*channel share*/ 1, n,
        c, h, w, module::isSign(getInput()),
        /*slope sign*/ 1, module::isSign(getOutput()),
        /*rshift num*/ 0, /*store mode*/ 1, 1,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::ReluOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  return 0;
}

void tpu::ReluOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
  auto input = getInput();
  auto output = getOutput();
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_g_info = LocalGenInterface::getGroupInfo(input, n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(input, n, c, h, w);
  int *input_shape = new int[4];
  module::getLocalShape(input, n_step, h_step, input_shape);
  int input_sign = module::isSign(input);
  int output_sign = module::isSign(output);

  auto relu_limit = getReluLimit().convertToDouble();
  if (!module::isUniformQuantized(output)) {
    BM1684::instance().dl_nodechip_relu_forward_local(
        in_g_info.out_addr, out_g_info.out_addr, input_shape, (float)relu_limit,
        BM1684::instance()->bdc_node);
  } else {
    BM1684::instance().dl_nodechip_prelu_forward_local_fix8b_v3(
        in_g_info.out_addr, out_g_info.out_addr,
        /*slope_local_offset*/ 0, in_g_info.out_addr,
        /*channel_shared*/ 1,
        /*shared_slope*/ 0, (uint32_t *)input_shape,
        /*st_by_fcway*/ 0, input_sign,
        /*not use slope_sign*/ 1, output_sign,
        /*rshift_bit*/ 0, int(relu_limit), BM1684::instance()->bdc_node);
  }
  delete[] input_shape;
}

uint32_t tpu::ReluOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::ReluOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::ReluOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
