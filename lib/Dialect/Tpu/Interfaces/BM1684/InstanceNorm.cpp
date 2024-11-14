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

void tpu::InstanceNormOp::codegen_global_bm1684() {
  auto input_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  uint64_t weight_addr = 0, bias_addr = 0;
  int input_shape[MAX_SHAPE_DIMS];
  int input_dims = module::getShape(getInput()).size();
  module::getGlobalShape(getInput(), input_shape);
  for (int i = 4; i < input_dims; ++i) {
    input_shape[3] *= input_shape[i];
  }
  int has_weight = !getWeight().getType().isa<NoneType>();
  int has_bias = !getBias().getType().isa<NoneType>();
  int affine = 0;
  if (has_weight) {
    weight_addr = module::getAddress(getWeight());
  }
  if (has_bias) {
    bias_addr = module::getAddress(getBias());
  }
  affine = has_weight | (has_bias << 1);
  BM1684::instance().dl_nodechip_group_norm(
      input_addr, weight_addr, bias_addr, out_addr, input_shape[0],
      input_shape[1], input_shape[2], input_shape[3],
      input_shape[1] /*group_nums*/, getEps().convertToDouble(), affine,
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

int64_t tpu::InstanceNormOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::InstanceNormOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                               local_sec_info_t &sec_info) {
  int input_shape[MAX_SHAPE_DIMS];
  int64_t depth = 1;
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  module::getLocalShape(getInput(), n_step, h_step, input_shape);
  if (out_g_info.type == GROUP_3D && module::getShape(getInput()).size() > 4) {
    int64_t n, c, h, w;
    module::getNCDHW(getInput(), n, c, depth, h, w, GROUP_3D);
  }
  auto input_g_info =
      LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  uint32_t weight_addr = 0, bias_addr = 0;
  int has_weight = !getWeight().getType().isa<NoneType>();
  int has_bias = !getBias().getType().isa<NoneType>();
  int affine = 0;
  if (has_weight) {
    auto weight_g_info =
        LocalGenInterface::getGroupInfo(getWeight(), n_step, h_step);
    weight_addr = weight_g_info.out_addr;
  }
  if (has_bias) {
    auto bias_g_info =
        LocalGenInterface::getGroupInfo(getBias(), n_step, h_step);
    bias_addr = bias_g_info.out_addr;
  }
  affine = has_weight | (has_bias << 1);
  BM1684::instance().dl_nodechip_group_norm_local(
      input_g_info.out_addr, weight_addr, bias_addr, out_g_info.buffer_addr,
      out_g_info.out_addr, input_shape[0], input_shape[1], input_shape[2],
      input_shape[3], depth, input_shape[1] /*group_nums*/,
      getEps().convertToDouble(), affine, BM1684::instance()->bdc_node);
}

uint32_t tpu::InstanceNormOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::InstanceNormOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::InstanceNormOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
