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
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;


void tpu::ReverseOp::codegen_global_cv18xx(int64_t layer_id) {
  gaddr_t ga_input = module::getAddress(input());
  gaddr_t ga_output = module::getAddress(output());
  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  int64_t _axis = axis();
  if (module::isUniformQuantized(output())) {
    cvi_backend_tg_reverse_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                  _axis, CVK_FMT_I8);
  } else {
    cvi_backend_tg_reverse_kernel(layer_id, ga_input, ga_output, n, c, h, w,
                                  _axis, CVK_FMT_BF16);
  }
}
