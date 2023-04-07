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

void tpu::PReluOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto bottom_global_addr = module::getAddress(getInput());
  auto slope_global_addr = module::getAddress(getSlope());
  auto top_global_addr = module::getAddress(getOutput());
  auto slope_num = module::getNumElements(getSlope());
  auto slope_type = module::getStorageType(getSlope());
  float slope_val = 0;
  int channel_shared = 0;
  if (slope_num == 1) {
    channel_shared = 1;
  }
  if (channel_shared) {
    if (slope_type.isF32()) {
      auto slopeOp = cast<top::WeightOp>(getSlope().getDefiningOp());
      auto slope_f32 = slopeOp.read<float>();
      float *slope_data = slope_f32->data();
      slope_val = (float)(*slope_data);
    } else {
      llvm_unreachable("Not Implemented");
    }
  }
  BM1684::instance().dl_nodechip_prelu_forward(
      bottom_global_addr, slope_global_addr, top_global_addr, slope_val,
      channel_shared, n, c, h, w, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}

int64_t tpu::PReluOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  int64_t n0, c0, h0, w0;
  module::getNCHW(getInput(), n0, c0, h0, w0);
  int c_per_npu_0 = ceiling_func(c0, BM1684::NPU_NUM);
  int64_t buffer_size =
      sizeof(float) * in_nslice * c_per_npu_0 * BM1684::eu_num(sizeof(float));
  int64_t tensor_size = sizeof(float) * in_nslice * c_per_npu_0 *
                        align_up(in_hslice * w0, BM1684::eu_num(sizeof(float)));
  buffer_size = 2 * buffer_size + tensor_size;
  return buffer_size;
}

void tpu::PReluOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                        local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto slope_gi = LocalGenInterface::getGroupInfo(getSlope(), n_step, h_step);
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0);
  n = out_gi.n_slice;
  h = out_gi.h_slice;

  uint32_t la_input = in_gi.out_addr;
  uint32_t la_output = out_gi.out_addr;
  uint32_t la_slope = slope_gi.out_addr;
  uint32_t la_buffer = out_gi.buffer_addr;
  auto slope_num = module::getNumElements(getSlope());
  auto slope_type = module::getStorageType(getSlope());

  int channel_shared = 0;
  if (slope_num == 1) {
    channel_shared = 1;
  }

  float slope_val = 0;
  if (channel_shared) {
    if (slope_type.isF32()) {
      auto slopeOp = cast<top::WeightOp>(getSlope().getDefiningOp());
      auto slope_f32 = slopeOp.read<float>();
      float *slope_data = slope_f32->data();
      slope_val = (float)(*slope_data);
    } else {
      llvm_unreachable("Not Implemented");
    }
  }

  if (module::isUniformQuantized(getOutput())) {
    int rshift_bit = getRshift();
    int upper_limit = -1;
    int input_sign = module::isSign(getInput());
    int slope_sign = module::isSign(getSlope());
    int output_sign = module::isSign(getOutput());

    uint32_t bottom_dim_fix8b[4] = {(uint32_t)n, (uint32_t)c, (uint32_t)h,
                                    (uint32_t)w};
    BM1684::instance().dl_nodechip_prelu_forward_local_fix8b_v3(
        la_input, la_output, la_slope, la_buffer, channel_shared, slope_val,
        bottom_dim_fix8b, 0, input_sign, slope_sign, output_sign, rshift_bit,
        upper_limit, (CMD_ID_NODE *)BM1684::instance().bdc_node);
  } else {
    int bottom_dim[4] = {(int)n, (int)c, (int)h, (int)w};
    BM1684::instance().dl_nodechip_prelu_forward_local_v2(
        la_input, la_output, la_slope, la_buffer, channel_shared, slope_val,
        bottom_dim, 0, (CMD_ID_NODE *)BM1684::instance().bdc_node);
  }
}

uint32_t tpu::PReluOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
int64_t tpu::PReluOp::get_fw_type_bm1684() {
  return -1;
}

int32_t tpu::PReluOp::dyn_codegen_local_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}