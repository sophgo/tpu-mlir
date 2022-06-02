//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::ReshapeOp::codegen_global_int8_bm1684() {
  auto in_addr = Module::getAddress(input());
  auto out_addr = Module::getAddress(output());
  if (in_addr == out_addr) {
    return;
  }
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  Module::getNCHW(input(), in, ic, ih, iw);
  Module::getNCHW(output(), on, oc, oh, ow);
  if (on != in) {
    llvm_unreachable("not support now");
  } else {
    int total_num = align_up(on, 4l) * oc * oh * ow;
    BM1684::instance().dl_nodechip_global_memcpy_ex(
        in_addr, out_addr, 1, total_num, total_num, DTYPE_FP32, DTYPE_FP32,
        total_num, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}
