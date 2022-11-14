//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// #include "tpu_mlir/Backend/BM168x/cv18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;


void tpu::SoftmaxOp::codegen_global_cv18xx(void* ctx, int64_t layer_id) {
   CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
   bool do_log = false;
   int axis = this->axis();
   gaddr_t ga_input = Module::getAddress(input());
   gaddr_t ga_output = Module::getAddress(output());
   gaddr_t exponential_table_data_lut_gaddr = Module::getAddress(table());
   gaddr_t exponential_slope_table_data_lut_gaddr = Module::getAddress(slope_table());
   gaddr_t reciprocal_table_data_lut_gaddr = Module::getAddress(reciprocal_table());
   gaddr_t reciprocal_mantissa_table_data_lut_gaddr = Module::getAddress(reciprocal_mantissa_table());
   std::vector<int64_t> shape;
   Module::getShapeVec(input(), shape);
   int dimension = shape.size();
   cvi_backend_tg_bf16_softmax_kernel(
      *backend_ctx, layer_id,
      ga_input,
      exponential_table_data_lut_gaddr, exponential_slope_table_data_lut_gaddr,
      reciprocal_table_data_lut_gaddr, reciprocal_mantissa_table_data_lut_gaddr,
      ga_output,
      shape.data(), axis, dimension, do_log);
}
