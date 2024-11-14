//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::ConvBwdWeightOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  ConvBwdWeight_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto &common = spec.common;
  common.groups = attr.groups;
  common.n = attr.n;
  common.ic = attr.ic;
  common.ih = attr.ih;
  common.iw = attr.iw;
  common.oc = attr.oc;
  common.oh = attr.oh;
  common.ow = attr.ow;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.sh = attr.sh;
  common.sw = attr.sw;
  common.pt = attr.pht;
  common.pb = attr.phb;
  common.pl = attr.pwl;
  common.pr = attr.pwr;
  common.has_bias = attr.has_bias;
  BM168x::call_global_func("backend_api_ConvBwdWeight_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::ConvBwdWeightOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return -1;
}

void tpu::ConvBwdWeightOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                                 int64_t h_step, int64_t d_step,
                                                 int64_t w_step,
                                                 group_type_t group_type,
                                                 local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  // auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto attr = parseParam();

  ConvBwdWeight_local_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto &common = spec.common;
  common.groups = attr.groups;
  common.n = attr.n;
  common.ic = attr.ic;
  common.ih = attr.ih;
  common.iw = attr.iw;
  common.oc = attr.oc;
  common.oh = attr.oh;
  common.ow = attr.ow;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.sh = attr.sh;
  common.sw = attr.sw;
  common.pt = attr.pht;
  common.pb = attr.phb;
  common.pl = attr.pwl;
  common.pr = attr.pwr;
  common.has_bias = attr.has_bias;
  BM168x::call_local_func("backend_api_ConvBwdWeight_local", &spec,
                          sizeof(spec), &sec_info, input_spec->data(),
                          output_spec->data());
}

// dynamic codegen
int64_t tpu::ConvBwdWeightOp::dyn_codegen_local_bm1684x(void *buffer) {
  return -1;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ConvBwdWeightOp::dyn_codegen_global_bm1684x(void *buffer) {
  return -1;
}

int64_t tpu::ConvBwdWeightOp::get_fw_type_bm1684x() { return FW_BMNET_CONV; }
