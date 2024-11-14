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

#define DEBUG_TYPE "tl_concat"

namespace tpu_mlir {
namespace backend {
// axis = 1
void cvi_backend_tl_concat(uint32_t layer_id, int *input_dim_c, int input_size,
                           int *output_dim, laddr_t *la_input,
                           laddr_t la_output, bool do_relu, int32_t *r_i8,
                           int32_t *m_i8, cvk_fmt_t fmt) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_concat:\n"
                                          "  layer_id %d, fmt %d\n",
                                          layer_id, fmt));
  for (int i = 0; i < input_size; i++) {
    LLVM_DEBUG(llvm::errs() << llvm::format(
                   "in %d (%d, %d, %d, %d), la_input:%d,\n", i, output_dim[0],
                   input_dim_c[i], output_dim[2], output_dim[3], la_input[i]));
  }

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  uint32_t out_csize_local =
      ALIGN(output_dim[2] * output_dim[3] * CV18xx::bytesize_of_fmt(fmt),
            CV18xx::EU_BYTES);
  uint32_t n = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t h = output_dim[2];
  uint32_t w = output_dim[3];

  uint32_t concat_c = 0;
  for (int i = 0; i < input_size; i++) {
    uint32_t out_offset = (concat_c / CV18xx::NPU_NUM) * out_csize_local;
    uint32_t out_addr = (concat_c % CV18xx::NPU_NUM) * CV18xx::LMEM_BYTES +
                        la_output + out_offset;
    int16_t multiplier = 1;
    if (m_i8 != nullptr && m_i8[i] != 0) {
      multiplier = static_cast<int16_t>(m_i8[i]);
    }
    uint8_t rshift = 0;
    if (r_i8 != nullptr) {
      rshift = static_cast<uint8_t>(r_i8[i]);
    }
    bool do_quant = true;
    if (do_relu == false && multiplier == 1 && rshift == 0) {
      do_quant = false;
    }

    cvk_tl_shape_t shape = CV18xx::tl_shape_t4(n, input_dim_c[i], h, w);
    cvk_tl_shape_t out_shape = CV18xx::tl_shape_t4(n, oc, h, w);

    cvk_tl_t tl_input = {};
    tl_input.start_address = la_input[i];
    tl_input.fmt = fmt;
    tl_input.shape = shape;
    tl_input.stride = CV18xx::tl_default_stride(shape, fmt, 1);

    cvk_tl_t tl_output = {};
    tl_output.start_address = out_addr;
    tl_output.fmt = fmt;
    tl_output.shape = shape;
    tl_output.stride = CV18xx::tl_default_stride(out_shape, fmt, 1);

    cvk_tdma_l2l_tensor_copy_param_t p10 = {0};
    p10.dst = &tl_output;
    p10.src = &tl_input;
    CV18xx::parallel_disable();
    CV18xx::tdma_l2l_tensor_copy(&p10);
    CV18xx::parallel_enable();

    if (do_quant) {
      tl_input.start_address = out_addr;
      cvk_tiu_mul_param_t p = {0};
      p.res_high = nullptr;
      p.res_low = &tl_output;
      p.a = &tl_output;
      if (fmt == CVK_FMT_BF16) {
        p.b_const.val = CV18xx::convert_fp32_to_bf16(1.0);
        p.rshift_bits = 0;
      } else {
        p.b_const.val = multiplier;
        p.rshift_bits = rshift;
      }
      p.b_const.is_signed = false;
      p.b_is_const = 1;
      p.layer_id = layer_id;
      p.relu_enable = do_relu ? 1 : 0;
      // offset start with CV18xx::NPU_NUM start, 0x1000 for hw limitation
      uint32_t step = 0x1000 - CV18xx::NPU_NUM;
      int align_up_c = align_up(tl_output.shape.c, step);
      int slice_nr = align_up_c / step;
      uint32_t in_csize_local =
          ALIGN(shape.h * shape.w * CV18xx::bytesize_of_fmt(fmt),
                CV18xx::EU_BYTES) *
          (step / CV18xx::NPU_NUM);
      for (int s = 0; s < slice_nr; s++) {
        cvk_tl_t _tl_output = {};
        _tl_output.start_address = tl_output.start_address + s * in_csize_local;
        _tl_output.fmt = fmt;
        _tl_output.shape = shape;
        _tl_output.shape.c = std::min(tl_output.shape.c - s * step, step);
        _tl_output.stride =
            CV18xx::tl_default_stride(shape, fmt, /*eu_aling=*/1);
        p.res_low = &_tl_output;
        p.a = &_tl_output;
        CV18xx::tiu_mul(&p);
      }
    }

    concat_c += input_dim_c[i];
  }
}

} // namespace backend
} // namespace tpu_mlir
