//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;


void tpu::SliceOp::codegen_global_cv18xx(void* ctx, int64_t layer_id) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;

  // prepare data
  std::vector<int64_t> i_s;
  std::vector<int64_t> o_s;
  std::vector<int> offset_4;
  std::vector<int> step_4;
  bool fusible;
  parseParam(i_s, o_s, offset_4, step_4, fusible);
  CVIKERNEL_FMT_E fmt;
  if (Quant::isUniformQuantized(output())) {
    fmt = CVK_FMT_I8;
  } else {
    fmt = CVK_FMT_BF16;
  }

  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  if (fusible == false) {
    cvi_backend_tg_crop_kernel(*backend_ctx, layer_id, ga_input, ga_output, i_s, o_s, offset_4, step_4, fmt);
  }
}
