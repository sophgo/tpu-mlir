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

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::CastOp::codegen_global_bm1684() {
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  if (qInput && !qOutput) {
    // int8 => fp32
    auto scale = module::getUniformQuantizedType(getInput()).getScale();
    BM1684::instance().dl_nodechip_global_int2float(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c,
        h, w, 1, STORAGE_MODE_4N_INT8,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getOutput()), n * c * h * w, scale,
        module::getAddress(getOutput()), BINARY_MUL, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node, 0);
  } else if (qOutput && !qInput) {
    // fp32 => int8
    auto scale = module::getUniformQuantizedType(getOutput()).getScale();
    BM1684::instance().dl_nodechip_const_binary(
        module::getAddress(getInput()), n * c * h * w, scale,
        module::getAddress(getInput()), BINARY_DIV, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node, 0);
    BM1684::instance().dl_nodechip_float2int8_v2(
        module::getAddress(getInput()), module::getAddress(getOutput()), n, c,
        h, w, 1, STORAGE_MODE_4N_INT8, ROUND_INF,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    dump();
    llvm_unreachable("CastOp type error");
  }
}

int64_t tpu::CastOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  int64_t local_buffer_size = 0;
  auto input_dtype = BM1684::getDataType(getInput());
  auto output_dtype = BM1684::getDataType(getOutput());
  if (input_dtype == DTYPE_FP32 &&
      (output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8)) {
    // double buffer
    local_buffer_size = in_lmem_bytes * 2;
  } else if (output_dtype == DTYPE_FP32 &&
             (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8)) {
    int64_t n, c, h, w;
    module::getNCHW(getInput(), n, c, h, w);
    int dsize = sizeof(float);
    local_buffer_size = ceiling_func(in_nslice, (int64_t)4) *
                        ceiling_func(c, BM1684::NPU_NUM) *
                        align_up(in_hslice * w, BM1684::eu_num(dsize)) * dsize;
  } else {
    llvm_unreachable("CastOp buffer type error");
  }
  return local_buffer_size;
}

void tpu::CastOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
  auto input_dtype = BM1684::getDataType(getInput());
  auto output_dtype = BM1684::getDataType(getOutput());
  auto input_group_info =
      LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto output_group_info = getGroupInfo(n_step, h_step);
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  uint32_t input_shape[4] = {(uint32_t)input_group_info.n_slice, (uint32_t)c,
                             (uint32_t)input_group_info.h_slice, (uint32_t)w};
  if (output_dtype == DTYPE_FP32 &&
      (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8)) {
    // fix8b => fp32
    auto scale = module::getUniformQuantizedType(getInput()).getScale();
    BM1684::instance().dl_tensor_int8_to_float_local_v2(
        input_group_info.out_addr, output_group_info.out_addr,
        output_group_info.buffer_addr, input_shape[0], input_shape[1],
        input_shape[2], input_shape[3], true, input_dtype == DTYPE_INT8 ? 1 : 0,
        STORAGE_MODE_4N_INT8, BM1684::instance().bdc_node);
    BM1684::instance().dl_nodechip_const_binary_local(
        output_group_info.out_addr, &input_shape[0], scale,
        output_group_info.out_addr, BINARY_MUL, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().bdc_node);
  } else if (input_dtype == DTYPE_FP32 &&
             (output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8)) {
    // fp32 => fix8b
    auto scale =
        (double)1 / module::getUniformQuantizedType(getOutput()).getScale();
    BM1684::instance().dl_nodechip_const_binary_local(
        input_group_info.out_addr, &input_shape[0], scale,
        input_group_info.out_addr, BINARY_MUL, 0, 0, 0,
        (CMD_ID_NODE *)BM1684::instance().bdc_node);
    BM1684::instance().dl_nodechip_float2int8_local_keep_input(
        input_group_info.out_addr, output_group_info.out_addr,
        output_group_info.buffer_addr, input_shape[0], input_shape[1],
        input_shape[2], input_shape[3], output_dtype == DTYPE_INT8 ? 1 : 0, 0,
        ROUND_INF, BM1684::instance().bdc_node);
  } else {
    llvm_unreachable("CastOp type error");
  }
}
