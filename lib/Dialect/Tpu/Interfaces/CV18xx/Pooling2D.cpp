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
void tpu::Pool2DOp::codegen_global_cv18xx(int64_t layer_id) {
  auto attr = parseParam();
  assert(!attr.do_relu);
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  if (getPoolMode() == tpu::PoolMode::Avg) {
    if (attr.pad_h || attr.pad_h_after || attr.pad_w || attr.pad_w_after) {
      // this->getOperation()->dump();
      assert(attr.count_include_pad &&
             "AvgPooling2d should be count_include_pad=true.");
    }
    if (module::isUniformQuantized(getOutput())) {
      cvi_backend_tg_fixed_avg_pooling_kernel(
          layer_id,  // layer_id,
          ga_input,  // input_data_gaddr,
          ga_output, // output_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          attr.do_relu,                    // int do_relu,
          (int8_t)getRshift().value(),     // int right_shift_width,
          (int8_t)getMultiplier().value(), // &threshold_x_quantized,
          true);
    } else {
      cvi_backend_tg_bf16_pooling_kernel(

          layer_id,   // layer_id,
          ga_input,   // input_data_gaddr,
          ga_output,  // output_data_gaddr,
          GA_INVALID, // index_data_gaddr,
          GA_INVALID, // o_findex_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          1,            // is_avg_pooling,
          0.0f,         // float avg_const,  // default(passing 0.0f) is 1/kh*kw
          attr.do_relu, // int do_relu,
          true);
    }
  } else if (getPoolMode() == tpu::PoolMode::Max) {
    if (module::isUniformQuantized(getOutput())) {
      cvi_backend_tg_fixed_max_pooling_kernel(

          layer_id,  // layer_id,
          ga_input,  // input_data_gaddr,
          ga_output, // output_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          attr.do_relu, // int do_relu,
          true);
    } else {
      cvi_backend_tg_bf16_pooling_kernel(

          layer_id,   // layer_id,
          ga_input,   // input_data_gaddr,
          ga_output,  // output_data_gaddr,
          GA_INVALID, // index_data_gaddr,
          GA_INVALID, // o_findex_data_gaddr,
          attr.n, attr.c, attr.ih, attr.iw, attr.kh, attr.kw, attr.pad_h,
          attr.pad_h_after, attr.pad_w, attr.pad_w_after, // pad (t, b, l, r)
          attr.sh, attr.sw,
          0,            // is_avg_pooling,
          0.0f,         // float avg_const,  // default(passing 0.0f) is 1/kh*kw
          attr.do_relu, // int do_relu,
          true);
    }
  } else {
    llvm_unreachable("Not supported now");
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::Pool2DOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::Pool2DOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                         int64_t d_step, int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info,
                                         int64_t layer_id) {
  auto attr = parseParam();

  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;

  int64_t pad_h_t = sec_info.h_idx == 0 ? attr.pad_h : 0;
  int64_t pad_h_b =
      sec_info.h_idx + sec_info.h_slice == attr.ih ? attr.pad_h_after : 0;

  if (getPoolMode() == tpu::PoolMode::Avg) {
    if (module::isUniformQuantized(getOutput())) {
      cvi_backend_tl_pooling(
          layer_id, la_input, la_output, sec_info.n_slice, attr.c,
          sec_info.h_slice, attr.iw, sec_info.out_n_slice, attr.c,
          sec_info.out_h_slice, attr.ow, attr.kh, attr.kw, attr.sh, attr.sw,
          pad_h_t, pad_h_b, attr.pad_w, attr.pad_w_after,
          true, /*is_avg_pooling,*/
          (int8_t)getRshift().value(), (int8_t)getMultiplier().value());
    } else {
      cvi_backend_tl_bf16_pooling(
          layer_id, la_input, la_output, sec_info.n_slice, attr.c,
          sec_info.h_slice, attr.iw, sec_info.out_n_slice, attr.c,
          sec_info.out_h_slice, attr.ow, attr.kh, attr.kw, attr.sh, attr.sw,
          pad_h_t, pad_h_b, attr.pad_w, attr.pad_w_after,
          true /*is_avg_pooling,*/);
    }
  } else if (getPoolMode() == tpu::PoolMode::Max) {
    if (module::isUniformQuantized(getOutput())) {
      int8_t rshift_i8 = 0;
      int8_t multiplier_i8 = 1;
      cvi_backend_tl_pooling(layer_id, la_input, la_output, sec_info.n_slice,
                             attr.c, sec_info.h_slice, attr.iw,
                             sec_info.out_n_slice, attr.c, sec_info.out_h_slice,
                             attr.ow, attr.kh, attr.kw, attr.sh, attr.sw,
                             pad_h_t, pad_h_b, attr.pad_w, attr.pad_w_after,
                             false, /*is_avg_pooling,*/
                             rshift_i8, multiplier_i8);
    } else {
      cvi_backend_tl_bf16_pooling(
          layer_id, la_input, la_output, sec_info.n_slice, attr.c,
          sec_info.h_slice, attr.iw, sec_info.out_n_slice, attr.c,
          sec_info.out_h_slice, attr.ow, attr.kh, attr.kw, attr.sh, attr.sw,
          pad_h_t, pad_h_b, attr.pad_w, attr.pad_w_after,
          false /*is_avg_pooling,*/);
    }
  } else {
    llvm_unreachable("Not supported now");
  }
  return;
}
