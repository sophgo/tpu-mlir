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
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::LRNOp::codegen_global_bm1684() {
  
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  if(false == module::isUniformQuantized(getInput())){
  BM1684::instance().dl_nodechip_lrn_forward_parallel(
      in_addr, out_addr, n, c, h, w, getAlpha().convertToDouble(), getSize(),
      getBeta().convertToDouble(), getBias().convertToDouble(),
      (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    int in_sign = module::isSign(getInput());
    BM1684::instance().dl_nodechip_lrn_fix8b_forward_parallel(
      in_addr, out_addr, n, c, h, w, in_sign, getAlpha().convertToDouble(), getSize(),
      getBeta().convertToDouble(), getBias().convertToDouble(), 1, 1,
      (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}
