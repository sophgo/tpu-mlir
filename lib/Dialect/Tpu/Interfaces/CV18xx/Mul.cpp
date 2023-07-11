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

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::MulOp::codegen_global_cv18xx(int64_t layer_id) {
  int input_num = getInputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w;
  std::vector<gaddr_t> ga_inputs;
  for (int i = 0; i < input_num; i++) {
    ga_inputs.emplace_back(module::getAddress(getInputs()[i]));
  }
  gaddr_t ga_output = module::getAddress(getOutput());

  bool do_early_stride = false;
  int early_stride_h = 0;
  int early_stride_w = 0;
  std::vector<int64_t> shape0(4, 1);
  std::vector<int64_t> shape1(4, 1);
  module::getNCHW(getInputs()[0], shape0[0], shape0[1], shape0[2], shape0[3],
                  false);
  module::getNCHW(getInputs()[1], shape1[0], shape1[1], shape1[2], shape1[3],
                  false);
  auto prod0 = std::accumulate(shape0.begin(), shape0.end(), 1,
                               std::multiplies<int64_t>());
  auto prod1 = std::accumulate(shape1.begin(), shape1.end(), 1,
                               std::multiplies<int64_t>());
  if (prod0 != prod1) {
    // only support broadcast right operand
    // TODO: support broadcast both operand
    if (prod0 < prod1) {
      std::reverse(ga_inputs.begin(), ga_inputs.end());
      std::swap(shape0, shape1);
    }
    if (module::isUniformQuantized(getOutput())) {
      int32_t multiplier_v = static_cast<int32_t>(this->getMultiplier());
      int32_t rshift_v = static_cast<int32_t>(this->getRshift());
      cvi_backend_tg_int8_bcast_mul_kernel(
          layer_id, ga_inputs[0], ga_inputs[1], ga_output, shape0[0], shape0[1],
          shape0[2], shape0[3], shape1[0], shape1[1], shape1[2], shape1[3],
          getDoRelu(), rshift_v, &multiplier_v);
    } else {
      cvi_backend_tg_bf16_bcast_mul_kernel(
          layer_id, ga_inputs[0], ga_inputs[1], ga_output, shape0[0], shape0[1],
          shape0[2], shape0[3], shape1[0], shape1[1], shape1[2], shape1[3],
          getDoRelu());
    }
  } else {
    n = shape0[0];
    c = shape0[1];
    h = shape0[2];
    w = shape0[3];
    if (module::isUniformQuantized(getOutput())) {
      int32_t multiplier_v = static_cast<int32_t>(this->getMultiplier());
      int32_t rshift_v = static_cast<int32_t>(this->getRshift());
      std::vector<int32_t> coeffs(input_num, 1);
      cvi_backend_tg_fixed_eltwise_mul_kernel(
          layer_id, ga_inputs.data(), ga_output, input_num, n, c, h, w,
          getDoRelu(), do_early_stride, early_stride_h, early_stride_w,
          rshift_v, &multiplier_v, coeffs.data());
    } else {
      std::vector<float> coeffs(input_num, 1.0);
      cvi_backend_tg_bf16_eltwise_mul_kernel(
          layer_id,         // layer_id
          ga_inputs.data(), // gaddr_t ga_input[]
          ga_output,        // gaddr_t ga_output
          input_num,        // int input_size
          n, c, h, w,
          getDoRelu(), // bool do_relu
          do_early_stride, early_stride_h, early_stride_w, coeffs.data());
    }
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::MulOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  if (!module::isUniformQuantized(getOutput())) {
    return 0;
  }
  int64_t n, c, h, w;
  auto vIn = getInputs()[0];
  module::getNCHW(vIn, n, c, h, w);
  n = in_nslice;
  h = in_hslice;
  auto fmt = CV18xx::getDataType(vIn);
  return CV18xx::lmem_woring_size({n, c, h, w}, 1, true, fmt);
}

void tpu::MulOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                      int64_t d_step, int64_t w_step,
                                      group_type_t group_type,
                                      local_sec_info_t &sec_info,
                                      int64_t layer_id) {
  int64_t input_num = getInputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w;
  auto shape = module::getShape(getInputs()[0]);
  module::getNCHW(shape, n, c, h, w);

  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in0_gi = LocalGenInterface::getGroupInfo(getInputs()[0], n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(getInputs()[1], n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  std::vector<laddr_t> la_input(2);
  la_input[0] = in0_gi.out_addr;
  la_input[1] = in1_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_working = gi.buffer_addr;

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 0;
  if (module::isUniformQuantized(getOutput())) {
    int32_t m_i32 = getMultiplier();
    int8_t rshift = getRshift();
    const int coeffs[2] = {1, 1};

    cvi_backend_tl_eltwise(layer_id, /*u32 layer_id,*/
                           la_input.data(), la_output, la_working, n, c, h, w,
                           input_num, op_code, rshift, 0,
                           /*m_i8*/ true,  /*use_default_coeff,*/
                           getDoRelu(), 0, /*relu_slope,*/
                           coeffs, m_i32, 0, 0, 0);
  } else {
    std::vector<float> coeffs(input_num, 1.0);
    cvi_backend_bf16_tl_eltwise(layer_id, /*u32 layer_id,*/
                                la_input.data(), la_output, n, c, h, w,
                                input_num, op_code, true, /*use_default_coeff,*/
                                getDoRelu(), 0,           /*relu_slope,*/
                                coeffs.data(), 0, 0, 0);
  }
  return;
}
