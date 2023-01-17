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

// using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ConcatOp::codegen_global_cv18xx(int64_t layer_id) {

  if (this->getOnlyMerge()) {
    return;
  }
  Operation *op = this->getOperation();
  int axis = this->getAxis();
  unsigned nInputs = op->getNumOperands();
  std::vector<gaddr_t> ga_inputs(nInputs);
  for (unsigned i = 0; i < nInputs; i++) {
    ga_inputs[i] = module::getAddress(this->getOperand(i));
  }
  gaddr_t ga_output = module::getAddress(getOutput());

  // prepare shape info
  std::vector<int32_t> axis_dims;
  for (unsigned i = 0; i < nInputs; i++) {
    auto shape = module::getShape(op->getOperand(i));
    axis_dims.push_back(shape[axis]);
  }

  std::vector<int32_t> output_dim;
  auto shape = module::getShape(this->getResult());
  int output_dim_size = shape.size();
  output_dim.assign(shape.begin(), shape.end());

  // prepare quant info
  std::vector<int32_t> rshift;
  std::vector<int32_t> m_i8_input;
  const int32_t *p_rshift = nullptr;
  const int32_t *p_m_i8 = nullptr;
  cvk_fmt_t dtype = CVK_FMT_BF16;
  if (module::isUniformQuantized(getOutput())) {
    dtype = CVK_FMT_I8;
    if (this->getRshifts().has_value() && this->getMultipliers().has_value()) {
      auto multiplier_v = module::getI64Array(this->getMultipliers().value());
      m_i8_input.assign(multiplier_v->begin(), multiplier_v->end());
      assert(m_i8_input.size() == nInputs);
      auto shift_v = module::getI64Array(this->getRshifts().value());
      rshift.assign(shift_v->begin(), shift_v->end());
      assert(rshift.size() == nInputs);
      p_rshift = rshift.data();
      p_m_i8 = m_i8_input.data();
    }
  }
  cvi_backend_tg_concat_kernel(
      layer_id, nInputs, ga_inputs.data(), ga_output, axis_dims.data(), axis,
      output_dim_size, output_dim.data(), getDoRelu(), p_rshift, p_m_i8, dtype);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ConcatOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::ConcatOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step, int64_t layer_id) {
  llvm_unreachable("Not supported now");
}
