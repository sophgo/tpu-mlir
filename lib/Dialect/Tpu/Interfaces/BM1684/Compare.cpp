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

void tpu::CompareOp::codegen_global_bm1684() {
  auto l_dim = module::getShape(getLhs()).size();
  auto r_dim = module::getShape(getRhs()).size();
  auto o_dim = module::getShape(getOutput()).size();
  int l_shape[MAX_SHAPE_DIMS], r_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getLhs(), l_shape);
  module::getGlobalShape(getRhs(), r_shape);
  if (!module::isUniformQuantized(getOutput())) {
    auto o_dtype = BM1684::getDataType(getOutput());
    int src_int32 = o_dtype == DTYPE_FP32 ? 0 : 1;
    auto gdma_format = BM1684::GDMA_VALUE_FORMAT_FLOAT32;
    BM1684::instance().dl_nodechip_broadcast_binary_full(
        module::getAddress(getLhs()), (uint32_t *)l_shape, l_dim,
        module::getAddress(getRhs()), (uint32_t *)r_shape, r_dim,
        module::getAddress(getOutput()), 0 /*buffer_addr*/,
        BM168x::compare_mode(getMode()), 0 /*relu*/, -1 /*relu limit*/,
        gdma_format, (CMD_ID_NODE *)BM1684::instance()->cmdid_node, src_int32);
  } else {
    auto l_dtype = BM1684::getDataType(getLhs());
    auto r_dtype = BM1684::getDataType(getRhs());
    auto o_dtype = BM1684::getDataType(getOutput());
    int is_int8[3] = {(l_dtype == DTYPE_INT8 || l_dtype == DTYPE_UINT8),
                      (r_dtype == DTYPE_INT8 || r_dtype == DTYPE_UINT8),
                      (o_dtype == DTYPE_INT8 || o_dtype == DTYPE_UINT8)};
    int is_sign[3] = {(l_dtype == DTYPE_INT8 || l_dtype == DTYPE_INT16),
                      (r_dtype == DTYPE_INT8 || r_dtype == DTYPE_INT16),
                      (o_dtype == DTYPE_INT8 || o_dtype == DTYPE_INT16)};

    BM1684::instance().dl_nodechip_broadcast_binary_fix8b_forward_parallel(
        module::getAddress(getLhs()), module::getAddress(getRhs()),
        module::getAddress(getOutput()), l_shape, r_shape, o_dim,
        module::isWeight(getLhs()), module::isWeight(getRhs()),
        BM168x::compare_mode(getMode()), 1, 1, 0,
        0, /*int scale_A, int scale_B, int rshift_A, int rshift_B*/
        is_int8, is_sign, 0, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

int64_t tpu::CompareOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::CompareOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                          local_sec_info_t &sec_info) {
  int shape_dim = module::getShape(getOutput()).size();
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto l_gi = LocalGenInterface::getGroupInfo(getLhs(), n_step, h_step);
  auto r_gi = LocalGenInterface::getGroupInfo(getRhs(), n_step, h_step);
  int l_shape[shape_dim], r_shape[shape_dim];
  module::getLocalShape(getLhs(), n_step, h_step, l_shape);
  module::getLocalShape(getRhs(), n_step, h_step, r_shape);
  if (module::isUniformQuantized(getOutput())) {
    auto l_dtype = BM1684::getDataType(getLhs());
    auto r_dtype = BM1684::getDataType(getRhs());
    auto o_dtype = BM1684::getDataType(getOutput());
    int is_int8[3] = {(l_dtype == DTYPE_INT8 || l_dtype == DTYPE_UINT8),
                      (r_dtype == DTYPE_INT8 || r_dtype == DTYPE_UINT8),
                      (o_dtype == DTYPE_INT8 || o_dtype == DTYPE_UINT8)};
    int is_sign[3] = {(l_dtype == DTYPE_INT8 || l_dtype == DTYPE_INT16),
                      (r_dtype == DTYPE_INT8 || r_dtype == DTYPE_INT16),
                      (o_dtype == DTYPE_INT8 || o_dtype == DTYPE_INT16)};
    BM1684::instance().dl_nodechip_broadcast_binary_fix8b_forward_local(
        l_gi.out_addr, r_gi.out_addr, out_gi.out_addr, out_gi.buffer_addr,
        l_shape, r_shape, shape_dim, module::isWeight(getLhs()),
        module::isWeight(getRhs()), BM168x::compare_mode(getMode()), 1, 1, 0, 0,
        is_int8, is_sign, 0, BM1684::instance()->bdc_node);
  } else {
    int l_stride[shape_dim], r_stride[shape_dim], o_stride[shape_dim],
        o_shape[shape_dim];
    module::getLocalShape(getOutput(), n_step, h_step, o_shape);
    module::get128BtyeAlignedStrideForNBit(l_stride, l_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(r_stride, r_shape, BM1684::NPU_NUM,
                                           32);
    module::get128BtyeAlignedStrideForNBit(o_stride, o_shape, BM1684::NPU_NUM,
                                           32);
    for (int i = 0; i < shape_dim; i++) {
      if (l_shape[i] != r_shape[i]) {
        if (l_shape[i] == 1)
          l_stride[i] = 0;
        if (r_shape[i] == 1)
          r_stride[i] = 0;
      }
    }
    BM1684::instance().dl_nodechip_broadcast_binary_local(
        l_gi.out_addr, l_shape, l_stride, r_gi.out_addr, r_shape, r_stride,
        out_gi.out_addr, o_stride, BM168x::compare_mode(getMode()), 0, -1,
        l_shape[1] > r_shape[1] ? r_gi.out_addr : l_gi.out_addr,
        BM1684::instance()->bdc_node);
  }
}

uint32_t tpu::CompareOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::CompareOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::CompareOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
