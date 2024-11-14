//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#define DEBUG_TYPE "tl_upsample"

namespace tpu_mlir {
namespace backend {

void cvi_backend_tl_upsample(uint32_t layer_id, laddr_t input_laddr,
                             laddr_t output_laddr, int input_n, int input_c,
                             int input_h, int input_w, int scale_h, int scale_w,
                             cvk_fmt_t fmt) {
  // input
  cvk_tl_shape_t tl_input_shape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t tl_input;
  tl_input.start_address = input_laddr;
  tl_input.fmt = fmt;
  tl_input.shape = tl_input_shape;
  tl_input.stride = CV18xx::tl_default_stride(tl_input_shape, fmt, 1);

  // output
  auto output_n = input_n;
  auto output_c = input_c;
  auto output_h = input_h * scale_h;
  auto output_w = input_w * scale_w;

  cvk_tl_shape_t tl_output_shape =
      CV18xx::tl_shape_t4(output_n, output_c, output_h, output_w);
  cvk_tl_t tl_output;
  tl_output.start_address = output_laddr;
  tl_output.fmt = fmt;
  tl_output.shape = tl_output_shape;
  tl_output.stride = CV18xx::tl_default_stride(tl_output_shape, fmt, 1);

  cvk_tiu_average_pooling_param_t param = {0};
  param.ofmap = &tl_output;
  param.ifmap = &tl_input;
  param.kh = scale_h;
  param.kw = scale_w;
  param.ins_h = scale_h - 1;
  param.ins_last_h = 0;
  param.ins_w = scale_w - 1;
  param.ins_last_w = 0;
  param.pad_top = scale_h - 1;
  param.pad_bottom = scale_h - 1;
  param.pad_left = scale_w - 1;
  param.pad_right = scale_w - 1;
  param.stride_h = 1;
  param.stride_w = 1;
  if (fmt == CVIKERNEL_FMT_E::CVK_FMT_BF16) {
    param.avg_pooling_const =
        CV18xx::convert_fp32_to_bf16((float)(scale_h * scale_w));
  } else {
    // refer it from the result, don't know the reason.
    param.avg_pooling_const = 1;
  }
  param.rshift_bits = 0;
  param.layer_id = layer_id;
  param.ins_val = 0;
  param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
  CV18xx::tiu_average_pooling(&param);
}

} // namespace backend
} // namespace tpu_mlir
