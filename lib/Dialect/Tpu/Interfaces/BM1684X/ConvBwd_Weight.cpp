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
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

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

void tpu::ConvBwdWeightOp::codegen_global_bm1684() {

}

void tpu::ConvBwdWeightOp::codegen_global_cv18xx(int64_t Layer_id) {

}


