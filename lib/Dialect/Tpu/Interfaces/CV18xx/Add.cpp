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

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::AddOp::codegen_global_cv18xx(int64_t layer_id) {
  int input_num = getInputs().size();
  int64_t n, c, h, w;
  std::vector<gaddr_t> ga_inputs;
  for (int i = 0; i < input_num; ++i) {
    ga_inputs.emplace_back(module::getAddress(getInputs()[i]));
  }

  gaddr_t ga_output = module::getAddress(getOutput());

  bool do_early_stride = false;
  int early_stride_h = 0;
  int early_stride_w = 0;
  if (getDoEarlyStride().has_value()) {
    do_early_stride = getDoEarlyStride().value();
    early_stride_h = getEarlyStrideH().value();
    early_stride_w = getEarlyStrideW().value();
  }

  auto coeffs_ = module::getF64Array(getCoeff(), input_num, 1);
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
    assert(input_num == 2);
    if (prod0 < prod1) {
      std::reverse(ga_inputs.begin(), ga_inputs.end());
      std::swap(shape0, shape1);
    }
    if (module::isUniformQuantized(getOutput())) {
      auto multiplier_v = module::getI64Array(getMultipliers(), input_num, 1);
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      int32_t rshift_int = static_cast<int32_t>(rshift_v->at(0));
      std::vector<int32_t> multiplier_int;
      for (int i = 0; i < input_num; ++i) {
        multiplier_int.emplace_back(multiplier_v->at(i));
      }
      std::vector<int> coeffs(input_num, 1);
      if (prod0 < prod1) {
        std::reverse(std::begin(multiplier_int), std::end(multiplier_int));
      }
      cvi_backend_tg_int8_bcast_add_kernel(
          layer_id, ga_inputs[0], ga_inputs[1], ga_output, shape0[0], shape0[1],
          shape0[2], shape0[3], shape1[0], shape1[1], shape1[2], shape1[3],
          getDoRelu(), rshift_int, multiplier_int.data());
    } else {
      cvi_backend_tg_bf16_bcast_add_kernel(
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
      auto multiplier_v = module::getI64Array(getMultipliers(), input_num, 1);
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      int32_t rshift_int = static_cast<int32_t>(rshift_v->at(0));
      std::vector<int32_t> multiplier_int;
      for (int i = 0; i < input_num; ++i) {
        multiplier_int.emplace_back(multiplier_v->at(i));
      }
      std::vector<int> coeffs;
      coeffs.assign(coeffs_->begin(), coeffs_->end());
      cvi_backend_tg_fixed_eltwise_add_kernel(
          layer_id, ga_inputs.data(), ga_output, input_num, n, c, h, w,
          getDoRelu(), do_early_stride, early_stride_h, early_stride_w,
          rshift_int, multiplier_int.data(), coeffs.data());
    } else {
      // TODO do_early_stride, coeffs
      std::vector<float> coeffs;
      coeffs.assign(coeffs_->begin(), coeffs_->end());
      cvi_backend_tg_bf16_eltwise_add_kernel(
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

int64_t tpu::AddOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
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

void tpu::AddOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
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

  // for case calc layergroup cycle
  if (la_output == la_working && la_output == 0) {
    la_working = CV18xx::LMEM_BYTES / 2;
  }

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  // early stride
  bool do_early_stride = false;
  int32_t early_stride_h = 0;
  int32_t early_stride_w = 0;
  if (getDoEarlyStride().has_value()) {
    do_early_stride = getDoEarlyStride().value();
    early_stride_h = getEarlyStrideH().value();
    early_stride_w = getEarlyStrideW().value();
  }

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 1;
  if (module::isUniformQuantized(getOutput())) {
    auto multiplier_v = module::getI64Array(getMultipliers(), input_num, 1);
    auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
    int8_t rshift_int = static_cast<int32_t>(rshift_v->at(0));
    std::vector<int8_t> multiplier_int;
    for (int i = 0; i < input_num; ++i) {
      multiplier_int.emplace_back(multiplier_v->at(i));
    }
    const int coeffs[2] = {1, 1};

    cvi_backend_tl_eltwise(
        layer_id, /*u32 layer_id,*/
        la_input.data(), la_output, la_working, n, c, h, w, input_num, op_code,
        rshift_int, multiplier_int.data(), true, /*use_default_coeff,*/
        getDoRelu(), 0,                          /*relu_slope,*/
        coeffs, 0, do_early_stride, early_stride_h, early_stride_w);
  } else {
    std::vector<float> coeffs;
    auto coeffs_ = module::getF64Array(getCoeff(), 2, 1);
    coeffs.assign(coeffs_->begin(), coeffs_->end());
    cvi_backend_bf16_tl_eltwise(layer_id, /*u32 layer_id,*/
                                la_input.data(), la_output, n, c, h, w,
                                input_num, op_code, true, /*use_default_coeff,*/
                                getDoRelu(), 0,           /*relu_slope,*/
                                coeffs.data(), do_early_stride, early_stride_h,
                                early_stride_w);
  }
  return;
}
