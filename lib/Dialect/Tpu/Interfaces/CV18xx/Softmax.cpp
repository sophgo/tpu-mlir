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
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::SoftmaxOp::codegen_global_cv18xx(int64_t layer_id) {
  bool do_log = getLog();
  int axis = this->getAxis();
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t exponential_table_data_lut_gaddr = module::getAddress(getTable());
  gaddr_t exponential_slope_table_data_lut_gaddr =
      module::getAddress(getSlopeTable());
  gaddr_t reciprocal_table_data_lut_gaddr =
      module::getAddress(getReciprocalTable());
  gaddr_t reciprocal_mantissa_table_data_lut_gaddr =
      module::getAddress(getReciprocalMantissaTable());
  std::vector<int64_t> shape;
  module::getShapeVec(getInput(), shape);
  int dimension = shape.size();
  cvi_backend_tg_bf16_softmax_kernel(
      layer_id, ga_input, exponential_table_data_lut_gaddr,
      exponential_slope_table_data_lut_gaddr, reciprocal_table_data_lut_gaddr,
      reciprocal_mantissa_table_data_lut_gaddr, ga_output, shape.data(), axis,
      dimension, do_log);
}
