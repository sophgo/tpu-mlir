//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

typedef enum {
  FcPerLayerShift = 0,
  FcPerLayerScale = 1,
  FcPerChannelScale = 2,
} FcQScale;

typedef struct {
  float perlayer_scale;
  int if_asymmetic;
  int weight_offset;
  int output_offset;
  int if_bias_float;
} FcQParams;

void tpu::MatMulOp::codegen_global_bm1684() {
  auto p = parseParam();
  int using_bias = p.with_bias ? 1 : 0;
  int if_relu = p.do_relu ? 1 : 0;
  double relu_upper_limit =
      p.do_relu ? p.relu_limit : 0.0; // no working in backend
  auto in_addr = module::getAddress(getInput());
  auto right_addr = module::getAddress(getRight());
  auto bias_addr = module::getAddress(getBias());
  auto out_addr = module::getAddress(getOutput());
  int batch_in = module::getNumElements(getInput()) / ((int)p.M * (int)p.K);
  int batch_right = module::getNumElements(getRight()) / ((int)p.K * (int)p.N);
  int in_shape[3] = {batch_in, (int)p.M, (int)p.K};
  int right_shape[3] = {batch_right, (int)p.K, (int)p.N};
  if (module::isUniformQuantized(getInput())) {
    int in_sign = module::isSign(getInput());
    int right_sign = module::isSign(getRight());
    int bias_sign = p.with_bias ? module::isSign(getBias()) : 0;
    int if_right_active = module::isWeight(getRight()) ? 0 : 1;
    auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
    assert(rshift_v->size() == 1);
    FcQParams quant_param{0, 0, 0, 0, 0};
    BM1684::instance().dl_nodechip_fc_fix8b_forward_parallel(
        in_addr, right_addr, bias_addr, out_addr, /*scale_addr*/ 0, p.M, p.K,
        p.N, /*transpose*/ 0, using_bias, in_sign, right_sign, bias_sign,
        rshift_v->at(0), /*res_16b*/ 0, if_relu, /*in_4N*/ 1, if_right_active,
        /*out_4N*/ 1, /*perlayer bias*/ 0, FcPerLayerShift, &quant_param,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_batch_matmul_forward_parallel_v2(
        in_addr, (const int *)in_shape, 3, right_addr, (const int *)right_shape,
        3, out_addr, bias_addr, using_bias,
        if_relu, // relu = 0
        (float)relu_upper_limit, NULL, NULL,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

// =========================================
// LocalGenInterface
// =========================================
int64_t tpu::MatMulOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::MatMulOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

uint32_t tpu::MatMulOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  auto p = parseParam();
  uint32_t fw_ir_length = 0;
  if (p.batch == 1) {
    fw_fc_layer_param_t fw_fc_layer_param = {0};
    ir_layer_info_t *fc_ir_layer_info = (ir_layer_info_t *)ir_layer_info;
    dynamic_common_ir_layer_info(fc_ir_layer_info, getInput(), getOutput());
    assign_fw_param((void *)&fw_fc_layer_param);
    fc_ir_layer_info->fw_layer_param_u.fw_fc_layer_param = fw_fc_layer_param;
    fw_ir_length += sizeof(fw_fc_layer_param_t);
  } else {
    ir_layer_info_t *batch_matmul_layer_info = (ir_layer_info_t *)ir_layer_info;
    dynamic_common_ir_layer_info(batch_matmul_layer_info, getInput(),
                                 getOutput());
    fw_batch_matmul_layer_param_t fw_batch_matmul_layer_param = {0};
    assign_fw_param((void *)&fw_batch_matmul_layer_param);
    batch_matmul_layer_info->fw_layer_param_u.fw_batch_matmul_layer_param =
        fw_batch_matmul_layer_param;
    fw_ir_length += sizeof(fw_batch_matmul_layer_param_t);
    if (batch_matmul_layer_info->extra_len > 0) {
      fw_ir_length += sizeof(uint32_t);
      fw_ir_length += batch_matmul_layer_info->extra_len;
    }
  }
  return fw_ir_length;
}

int64_t tpu::MatMulOp::get_fw_type_bm1684() {
  auto p = parseParam();
  return (p.batch != 1 ? FW_BMNET_BATCH_MATMUL : FW_BMNET_FC);
}

// ======================================
// Dynamic LocalGenInterface
// ======================================

int32_t tpu::MatMulOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
