//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

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
    if (this->getRshifts().has_value() && getMultipliers().has_value()) {
      auto multiplier_v = module::getI64Array(getMultipliers().value());
      m_i8_input.assign(multiplier_v->begin(), multiplier_v->end());
      assert(m_i8_input.size() == nInputs);
      auto shift_v = module::getI64Array(getRshifts().value());
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
  return 0;
}

void tpu::ConcatOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info,
                                         int64_t layer_id) {
  int axis = getAxis();
  auto nInputs = getNumOperands();
  std::vector<int32_t> axis_dims;
  std::vector<laddr_t> la_input;
  for (auto in : getInputs()) {
    auto in_gi = LocalGenInterface::getGroupInfo(in, n_step, h_step);
    la_input.push_back(in_gi.out_addr);
    auto shape = module::getShape(in);
    axis_dims.push_back(shape[axis]);
  }

  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  laddr_t la_output = out_gi.out_addr;

  std::vector<int32_t> output_dim;
  auto shape = module::getShape(getOutput());
  output_dim.assign(shape.begin(), shape.end());
  for (uint32_t i = output_dim.size(); i < 4; i++) {
    output_dim.push_back(1); // fill to 4 dim
  }
  output_dim[0] = sec_info.out_n_slice;
  output_dim[2] = sec_info.out_h_slice;

  // prepare quant info
  std::vector<int32_t> m_i8_array;
  std::vector<int32_t> r_i8_array;
  int32_t *r_i8 = nullptr;
  int32_t *m_i8 = nullptr;
  cvk_fmt_t dtype = CVK_FMT_BF16;
  if (module::isUniformQuantized(getOutput())) {
    dtype = CVK_FMT_I8;
    if (getRshifts().has_value() && getMultipliers().has_value()) {
      auto multiplier_v = module::getI64Array(getMultipliers().value());
      auto shift_v = module::getI64Array(getRshifts().value());
      assert(multiplier_v->size() == shift_v->size());
      m_i8_array.assign(multiplier_v->begin(), multiplier_v->end());
      r_i8_array.assign(shift_v->begin(), shift_v->end());
      assert(m_i8_array.size() == nInputs);
      assert(r_i8_array.size() == nInputs);
      r_i8 = r_i8_array.data();
      m_i8 = m_i8_array.data();
    }
  }
  cvi_backend_tl_concat(layer_id, axis_dims.data(), nInputs, output_dim.data(),
                        la_input.data(), la_output, getDoRelu(), r_i8, m_i8,
                        dtype);
}
