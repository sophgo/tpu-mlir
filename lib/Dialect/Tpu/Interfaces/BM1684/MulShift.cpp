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
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::MulShiftOp::codegen_global_bm1684() {
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  int in_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInput(), in_shape);
  int sign[3];
  sign[0] = module::isSign(getInput());
  sign[1] = 1;
  sign[2] = module::isSign(getOutput());
  auto last_dim = in_shape[3];
  auto input_size = module::getShape(getInput()).size();
  for (int i = 4; i < input_size; i++) {
    last_dim *= in_shape[i];
  }
  BM1684::instance().dl_nodechip_mulshift_fix8b_forward(
      in_addr, out_addr, in_shape[0], in_shape[1], in_shape[2], last_dim,
      getMultiplier(), getRshift(), sign[0], sign[1], sign[2],
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

int64_t tpu::MulShiftOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (module::isUniformQuantized(getInput())) {
    int64_t buffer_size =
        ceiling_func(in_nslice, (int64_t)2) * ceiling_func(c, BM1684::NPU_NUM) *
        align_up(in_hslice * w, BM1684::eu_num(sizeof(float))) * sizeof(int);
    return buffer_size;
  }
  return 0;
}

void tpu::MulShiftOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                           local_sec_info_t &sec_info) {
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  int out_shape[MAX_SHAPE_DIMS];
  module::getLocalShape(getOutput(), n_step, h_step, out_shape);
  int sign[3];
  sign[0] = module::isSign(getInput());
  sign[1] = 1;
  sign[2] = module::isSign(getOutput());
  BM1684::instance().dl_nodechip_mulshift_fix8b_forward_local(
      in_g_info.out_addr, gi.buffer_addr, gi.out_addr, out_shape[0],
      out_shape[1], out_shape[2], out_shape[3], getMultiplier(), getRshift(),
      sign[0], sign[1], sign[2], (CMD_ID_NODE *)BM1684::instance()->bdc_node);
}

uint32_t tpu::MulShiftOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  GLOBAL_IR_COMMON(mulshift);
}

int64_t tpu::MulShiftOp::get_fw_type_bm1684() { return FW_BMNET_MULSHIFT; }

int32_t tpu::MulShiftOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int32_t fw_ir_length = 0;
  IR_PARAM_COMMON(mulshift);
  // input tensor
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getInput());
  // output
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getOutput());
  // imm buffer
  int using_immbuf = module::isUniformQuantized(getInput()) &&
                     module::isSign(getInput()) && !module::isSign(getOutput());
  if (using_immbuf) {
    dynamic_push_back_local_buffer(layer_info->ir_tensor_info_v, 0,
                                   getOutput());
  }
  // compute fw ir info length for input, buffer and output
  fw_ir_length += (sizeof(uint32_t) + 2 * sizeof(uint32_t) +
                   using_immbuf * sizeof(uint32_t));
  // add fw ir length for output consumer number
  fw_ir_length += sizeof(uint32_t);
  return fw_ir_length;
}
