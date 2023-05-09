//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "tl_quant"

namespace tpu_mlir {
namespace backend {
// for layer group
void cvi_backend_tl_quant(uint32_t layer_id, laddr_t la_input,
                          laddr_t la_output, laddr_t la_working, cvk_fmt_t from,
                          cvk_fmt_t to, float const_scale, int n, int c, int h,
                          int w, bool bExtraInput) {

  CV18xx::set_layer_id(layer_id);
  LLVM_DEBUG(llvm::errs() << "    Quant  : nchw = (" << n << "," << c << ","
                          << h << "," << w << ")"
                          << "\n                     "
                          << "la_i = " << la_input << ", la_o = " << la_output
                          << ", bExtraInput: " << bExtraInput << "\n";);
  assert((from == CVK_FMT_I8 || from == CVK_FMT_U8 || from == CVK_FMT_BF16) &&
         "`from` only support int8/bf16");
  assert((to == CVK_FMT_I8 || to == CVK_FMT_BF16) &&
         "`to` only support int8/bf16");
  assert((from != to) && "`from` and `to` not equal");

  int is_dequant =
      ((from == CVK_FMT_I8 || from == CVK_FMT_U8) && to == CVK_FMT_BF16);

  cvk_tl_t *tl_input = new cvk_tl_t;
  cvk_tl_t *tl_output = new cvk_tl_t;
  memset(tl_input, 0, sizeof(*tl_input));
  memset(tl_output, 0, sizeof(*tl_output));

  tl_input->start_address = la_input;
  tl_input->fmt = from;
  tl_input->shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_input->stride =
      CV18xx::tl_default_stride(tl_input->shape, tl_input->fmt, /*eu_align=*/1);

  tl_output->start_address = la_output;
  tl_output->fmt = to;
  tl_output->shape = CV18xx::tl_shape_t4(n, c, h, w);
  tl_output->stride = CV18xx::tl_default_stride(tl_output->shape,
                                                tl_output->fmt, /*eu_align=*/1);

  // NOTICE: make sure tdma order before mul
  // compute
  if (is_dequant) {
    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.src = tl_input;
    p1.dst = tl_output;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_copy(&p1);
    CV18xx::parallel_enable();

    // move to high accurcy to calculate quant/dequant
    cvk_tiu_mul_param_t p = {0};
    p.res_high = NULL;
    p.res_low = tl_output;
    p.a = tl_output;
    p.b_is_const = 1;
    p.b_const.val = CV18xx::convert_fp32_to_bf16(const_scale);
    p.relu_enable = 0;
    p.layer_id = layer_id;

    uint32_t step = 0x1000 - CV18xx::NPU_NUM;
    int slice_nr = align_up(tl_output->shape.c, step) / step;
    uint32_t in_csize_local = ALIGN(tl_output->shape.h * tl_output->shape.w *
                                        CV18xx::bytesize_of_fmt(tl_output->fmt),
                                    CV18xx::EU_BYTES) *
                              (step / CV18xx::NPU_NUM);
    for (int s = 0; s < slice_nr; s++) {
      cvk_tl_t _tl_output = {};
      _tl_output.start_address = tl_output->start_address + s * in_csize_local;
      _tl_output.fmt = tl_output->fmt;
      _tl_output.shape = tl_output->shape;
      _tl_output.shape.c = std::min(tl_output->shape.c - s * step, step);
      _tl_output.stride = CV18xx::tl_default_stride(
          tl_output->shape, tl_output->fmt, /*eu_aling=*/1);
      p.res_low = &_tl_output;
      p.a = &_tl_output;
      CV18xx::tiu_mul(&p);
    }

  } else {
    cvk_tl_t *tl_working = new cvk_tl_t;
    memcpy(tl_working, tl_input, sizeof(cvk_tl_t));

    if (bExtraInput)
      tl_working->start_address = la_working;
    // quant, bf16->int8
    // move to high accurcy to calculate quant/dequant
    cvk_tiu_mul_param_t p = {0};
    p.res_high = NULL;
    p.res_low = tl_working;
    p.a = tl_input;
    p.b_is_const = 1;
    p.b_const.val = CV18xx::convert_fp32_to_bf16(const_scale);
    p.relu_enable = 0;
    p.layer_id = layer_id;

    CV18xx::tiu_mul(&p);

    // leverage l2l to implement bf16->int8
    cvk_tdma_l2l_tensor_copy_param_t p1 = {0};
    p1.src = tl_working;
    p1.dst = tl_output;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_copy(&p1);
    CV18xx::parallel_enable();

    delete tl_working;
  }

  delete tl_output;
  delete tl_input;
}

} // namespace backend
} // namespace tpu_mlir
