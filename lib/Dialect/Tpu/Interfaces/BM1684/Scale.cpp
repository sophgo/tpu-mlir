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

void tpu::ScaleOp::codegen_global_bm1684() {
  auto input_addr = module::getAddress(getInput());
  auto output_addr = module::getAddress(getOutput());
  auto relu_limit = getReluLimit().convertToDouble();
  int input_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInput(), input_shape);
  int input_dims = module::getShape(getInput()).size();
  for (int i = 4; i < input_dims; ++i) {
    input_shape[3] *= input_shape[i];
  }
  int has_scale = !getScale().getType().isa<NoneType>();
  int has_bias = !getBias().getType().isa<NoneType>();
  uint64_t scale_addr = 0, bias_addr = 0, lshift_addr = 0;
  int scale_sign = 0, bias_sign = 0;
  int input_sign = module::isSign(getInput());
  if (has_scale) {
    scale_addr = module::getAddress(getScale());
    scale_sign = module::isSign(getScale());
  }
  if (has_bias) {
    bias_addr = module::getAddress(getBias());
    bias_sign = module::isSign(getBias());
  }
  if (!module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_scale_forward(
        input_addr, scale_addr, bias_addr, output_addr, input_shape[0],
        input_shape[1], input_shape[2], input_shape[3], 1, 1, 1, getDoRelu(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    lshift_addr = module::getAddress(getLshift());
    BM1684::instance()
        .dl_nodechip_bnscale_forward_parallel_fix8b_with_src_storage_mode(
            input_addr, output_addr, scale_addr, bias_addr, lshift_addr,
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            input_sign, scale_sign, bias_sign, 2, getDoRelu(), int(relu_limit),
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ScaleOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  int64_t buffer_size = 0;
  auto out_type = module::getStorageType(getOutput());
  if (!out_type.isa<FloatType>()) {
    int64_t n, c, h, w;
    module::getNCHW(getInput(), n, c, h, w);
    int c_per_npu = ceiling_func(c, BM1684::NPU_NUM);
    buffer_size = sizeof(int) * ceiling_func(in_nslice, 2) * c_per_npu *
                  align_up(in_hslice * w, BM1684::eu_num(sizeof(float)));
  }
  return buffer_size;
}

void tpu::ScaleOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                        local_sec_info_t &sec_info) {
  int input_shape[MAX_SHAPE_DIMS];
  int scale_shape[MAX_SHAPE_DIMS];
  int bias_sign = 0;
  int bias_is_coeff = 1;
  int64_t depth = 1;
  uint32_t bias_addr = 0;
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto relu_limit = getReluLimit().convertToDouble();
  module::getLocalShape(getInput(), n_step, h_step, input_shape);
  module::getLocalShape(getScale(), n_step, h_step, scale_shape);
  if (out_g_info.type == GROUP_3D && module::getShape(getInput()).size() > 4) {
    int64_t n, c, h, w;
    module::getNCDHW(getInput(), n, c, depth, h, w, GROUP_3D);
  }
  auto input_g_info =
      LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto scale_g_info =
      LocalGenInterface::getGroupInfo(getScale(), n_step, h_step);
  int has_bias = !getBias().getType().isa<NoneType>();
  if (has_bias) {
    auto bias_g_info =
        LocalGenInterface::getGroupInfo(getBias(), n_step, h_step);
    bias_addr = bias_g_info.out_addr;
    bias_sign = module::isSign(getBias());
    bias_is_coeff = isa_and_nonnull<tpu::LoadOp>(getBias().getDefiningOp());
  }
  if (module::isUniformQuantized(getOutput())) {
    input_shape[1] *= depth;
    int input_sign = module::isSign(getInput());
    int scale_sign = module::isSign(getScale());
    auto lshift_g_info =
        LocalGenInterface::getGroupInfo(getLshift(), n_step, h_step);
    BM1684::instance().dl_nodechip_bnscale_fix8b_forward_local(
        input_g_info.out_addr, out_g_info.buffer_addr, scale_g_info.out_addr,
        bias_addr, lshift_g_info.out_addr, out_g_info.out_addr, input_shape[0],
        input_shape[1], input_shape[2], input_shape[3], input_sign, scale_sign,
        bias_sign, 3, 3, 3, getDoRelu(), int(relu_limit),
        BM1684::instance()->bdc_node);
  } else {
    input_shape[0] *= depth;
    int scale_is_coeff =
        isa_and_nonnull<tpu::LoadOp>(getScale().getDefiningOp());
    BM1684::instance().dl_nodechip_scale_forward_local(
        input_g_info.out_addr, scale_g_info.out_addr, bias_addr,
        out_g_info.out_addr, input_shape, scale_shape, has_bias, getDoRelu(),
        relu_limit, scale_is_coeff, bias_is_coeff,
        BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::ScaleOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::ScaleOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::ScaleOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
