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

void tpu::WhereOp::codegen_global_bm1684() {
  select_common_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.sel0_is_const = getXIsConst();
  spec.sel1_is_const = getYIsConst();
  spec.sel0_const_val = getXConstVal().convertToDouble();
  spec.sel1_const_val = getYConstVal().convertToDouble();
  uint32_t *cond_shape = new uint32_t[MAX_SHAPE_DIMS];
  uint32_t *out_shape = new uint32_t[MAX_SHAPE_DIMS];
  for (int i = 0; i < MAX_SHAPE_DIMS; i++) {
    cond_shape[i] = 1;
    out_shape[i] = 1;
  }
  int cond_sign = module::isSign(getCond());
  for (auto v : llvm::enumerate(module::getShape(getCond())))
    cond_shape[v.index()] = (uint32_t)v.value();
  for (auto v : llvm::enumerate(module::getShape(getOutput())))
    out_shape[v.index()] = (uint32_t)v.value();
  if (module::isUniformQuantized(getOutput())) {
    int64_t out_n, out_c, out_h, out_w;
    int64_t cond_n, cond_c, cond_h, cond_w;
    module::getNCHW(getOutput(), out_n, out_c, out_h, out_w);
    module::getNCHW(getCond(), cond_n, cond_c, cond_h, cond_w);
    BM1684::instance().dl_nodechip_select_fix8b(
        module::getAddress(getCond()), module::getAddress(getTbrn()),
        module::getAddress(getFbrn()), module::getAddress(getOutput()), 0,
        out_n, out_c, out_h, out_w, out_n / cond_n, out_c / cond_c,
        out_h / cond_h, out_w / cond_h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cond_sign,
        spec.sel0_is_const, spec.sel0_const_val, cond_sign, 1, 0,
        spec.sel1_is_const, spec.sel1_const_val, cond_sign, 1, 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_select_all(
        module::getAddress(getCond()), cond_shape,
        module::getAddress(getTbrn()), spec.sel0_is_const, spec.sel0_const_val,
        module::getAddress(getFbrn()), spec.sel1_is_const, spec.sel1_const_val,
        module::getAddress(getOutput()), out_shape, 0, 0.0, cond_sign,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
  delete[] cond_shape;
  delete[] out_shape;
}

int64_t tpu::WhereOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::WhereOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                        local_sec_info_t &sec_info) {
  select_common_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.sel0_is_const = getXIsConst();
  spec.sel1_is_const = getYIsConst();
  spec.sel0_const_val = getXConstVal().convertToDouble();
  spec.sel1_const_val = getYConstVal().convertToDouble();
  uint32_t *cond_shape = new uint32_t[MAX_SHAPE_DIMS];
  uint32_t *out_shape = new uint32_t[MAX_SHAPE_DIMS];
  int cond_sign = module::isSign(getCond());
  auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto cond_gi = LocalGenInterface::getGroupInfo(getCond(), n_step, h_step);
  auto tbrn_gi = LocalGenInterface::getGroupInfo(getTbrn(), n_step, h_step);
  auto fbrn_gi = LocalGenInterface::getGroupInfo(getFbrn(), n_step, h_step);
  for (auto v : llvm::enumerate(module::getShape(getCond())))
    cond_shape[v.index()] = (uint32_t)v.value();
  for (auto v : llvm::enumerate(module::getShape(getOutput())))
    out_shape[v.index()] = (uint32_t)v.value();
  if (module::isUniformQuantized(getOutput())) {
    int64_t out_n, out_c, out_h, out_w;
    int64_t cond_n, cond_c, cond_h, cond_w;
    module::getNCHW(getOutput(), out_n, out_c, out_h, out_w);
    module::getNCHW(getCond(), cond_n, cond_c, cond_h, cond_w);
    BM1684::instance().dl_nodechip_select_fix8b_local(
        out_gi.out_addr, cond_shape, out_gi.buffer_addr, tbrn_gi.out_addr,
        spec.sel0_is_const, spec.sel0_const_val, fbrn_gi.out_addr,
        spec.sel1_is_const, spec.sel1_const_val, out_gi.out_addr, cond_shape,
        cond_sign, cond_sign, cond_sign, 1, 0, 1, 0, 0, 0.0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_select_local(
        cond_gi.out_addr, cond_shape, out_gi.buffer_addr, tbrn_gi.out_addr,
        spec.sel0_is_const, spec.sel0_const_val, fbrn_gi.out_addr,
        spec.sel1_is_const, spec.sel1_const_val, out_gi.out_addr, out_shape, 0,
        0.0, cond_sign, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
  delete[] cond_shape;
  delete[] out_shape;
}

uint32_t tpu::WhereOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::WhereOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::WhereOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
